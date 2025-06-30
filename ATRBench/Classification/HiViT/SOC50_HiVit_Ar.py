import sys
import torch
import numpy as np
import argparse
import torch.nn as nn
import torchvision.transforms as transforms
from utils.DataLoad import load_data
from utils.TrainTest import model_test
from model.HiVit import HiViT_base
import os
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import Subset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import itertools
from tqdm.auto import tqdm
import torch.hub
from torch.optim.adamw import AdamW
from torch.cuda import amp

def init_distributed_mode(args):
    """初始化 DDP 环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    else:
        print('不使用分布式训练')
        args.distributed = False
        args.rank = 0
        args.gpu = 0
        return

    args.distributed = True
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print(f'| DDP 初始化 (Rank {args.rank}): {args.dist_url}', flush=True)
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    dist.barrier()

def cleanup():
    """清理 DDP 环境"""
    if dist.is_initialized():
        dist.destroy_process_group()

def reduce_value(value, average=True):
    """在所有 DDP 进程中同步和平均一个值"""
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size < 2:
        return value
    with torch.no_grad():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        if average:
            value /= world_size
        return value

#参数设置

def parameter_setting():
    """参数设置函数"""
    parser = argparse.ArgumentParser(description='HiViT UpUp')
    
    # --- 基本参数 ---
    parser.add_argument('--data_path', type=str, default='../../datasets/SOC_50classes/', help='数据集路径')
    parser.add_argument('--epochs', type=int, default=256, help='训练总轮数')
    parser.add_argument('--classes', type=int, default=50, help='类别数量')
    parser.add_argument('--batch_size', type=int, default=188, help='单个GPU的批处理大小')
    parser.add_argument('--workers', type=int, default=8, help='数据加载的工作线程数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--img_size', type=int, default=224, help='输入图像尺寸')
    
    # --- 学习率与优化器参数 ---
    parser.add_argument('--lr', type=float, default=5e-4, help='学习率')
    parser.add_argument('--warmup_epochs', type=int, default=20, help='预热轮数')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='最低学习率')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='权重衰减')
    
    # --- 正则化与早停参数 ---
    parser.add_argument('--patience', type=int, default=30, help='早停: 验证集性能无提升的等待轮数')
    parser.add_argument('--overfit_gap_threshold', type=float, default=20.0, help='早停: 训练与验证准确率差距阈值')
    parser.add_argument('--overfit_check_epoch', type=int, default=20, help='开始检查过拟合的轮数')
    # --- 训练稳定性参数 ---
    parser.add_argument('--clip_grad', type=float, default=1.0, help='梯度裁剪阈值 (<=0 表示不裁剪)')

    # --- 预训练与数据增强参数 ---
    parser.add_argument('--pretrained_weights_url', type=str,
                        default='',
                        help='预训练权重（留空则使用本地文件）')
    
    # --- DDP 相关参数 ---
    parser.add_argument('--dist_url', default='env://', help='DDP 使用的 URL')
    parser.add_argument('--resume', default='', type=str, help='要恢复训练的检查点路径')
    
    # --- 测试模式参数 ---
    parser.add_argument('--test_only', action='store_true', help='仅运行测试模式')
    parser.add_argument('--test_checkpoint', type=str, default='', help='用于测试的 checkpoint 路径')

    args = parser.parse_args()
    return args

def get_data_transforms(img_size=224):
    """获取包含强力数据增强的变换"""
    # 验证和测试集只做最基础的变换
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 训练集使用激进的自动数据增强策略
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3)),  # 随机擦除
    ])
    return train_transform, val_transform

#核心训练与评估逻辑

def load_pretrained_weights(model, url, num_classes):
    """加载预训练权重，并智能处理分类头不匹配的问题"""
    try:
        print(f"正在从 {url} 加载预训练权重...")
        checkpoint = torch.hub.load_state_dict_from_url(url, map_location='cpu', check_hash=True)
        
        # 官方权重文件可能包含 'model' 键
        if 'model' in checkpoint:
            checkpoint = checkpoint['model']

        # 移除与分类相关的键
        state_dict = {k: v for k, v in checkpoint.items() if 'head' not in k}
        
        # 加载权重
        msg = model.load_state_dict(state_dict, strict=False)
        print("预训练权重加载完毕。忽略的键:", msg.missing_keys)
        
    except Exception as e:
        print(f"加载预训练权重失败: {e}。将从零开始训练。")


def train_one_epoch(model, data_loader, optimizer, criterion, device, epoch, scaler, args):
    """单轮训练循环，包含 DDP 同步和 AMP"""
    model.train()
    sampler = data_loader.sampler
    if args.distributed:
        sampler.set_epoch(epoch)

    # 准备记录每个进程的指标
    loss_meter = torch.tensor(0.0, device=device)
    correct_meter = torch.tensor(0.0, device=device)
    
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch}/{args.epochs}", disable=(args.rank != 0), dynamic_ncols=True)

    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        
        with amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        if args.clip_grad > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        scaler.step(optimizer)
        scaler.update()

        # 记录当前批次的指标
        loss_meter += loss.detach()
        correct_meter += (outputs.argmax(1) == labels).sum()

    # DDP - 同步所有GPU的指标
    total_loss = reduce_value(loss_meter, average=False)
    total_correct = reduce_value(correct_meter, average=False)
    
    avg_loss = total_loss.item() / len(sampler.dataset)
    accuracy = 100. * total_correct.item() / len(sampler.dataset)
    return avg_loss, accuracy

@torch.no_grad()
def evaluate(model, data_loader, criterion, device, args, desc=""):
    """评估函数，用于验证集和测试集，并实时显示精度"""
    model.eval()
    correct_meter = torch.tensor(0.0, device=device)
    total_samples_meter = torch.tensor(0.0, device=device)

    # 准备进度条
    progress_bar = tqdm(data_loader, desc=desc, disable=(args.rank != 0), dynamic_ncols=True)
    
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with amp.autocast():
            outputs = model(inputs)
        
        correct_meter += (outputs.argmax(1) == labels).sum()
        total_samples_meter += inputs.size(0)

        # 在 DDP 环境下，同步各个进程的中间结果以计算实时准确率
        if args.distributed:
            # 创建张量副本进行 all_reduce 操作，避免修改原始累加值
            synced_correct = reduce_value(correct_meter.clone(), average=False)
            synced_total = reduce_value(total_samples_meter.clone(), average=False)
        else:
            synced_correct = correct_meter
            synced_total = total_samples_meter
        
        # 只在主进程更新进度条描述
        if args.rank == 0:
            current_acc = 100. * synced_correct.item() / synced_total.item() if synced_total.item() > 0 else 0.0
            progress_bar.set_description(f"{desc} (OA: {current_acc:.2f}%)")

    # 循环结束后，进行最终的同步，计算最终准确率
    if args.distributed:
        total_correct = synced_correct
        total_samples = len(data_loader.sampler.dataset)
    else:
        total_correct = correct_meter
        total_samples = len(data_loader.dataset)
        
    accuracy = 100. * total_correct.item() / total_samples
    return accuracy

def main():
    """主执行函数"""
    arg = parameter_setting()
    init_distributed_mode(arg)
    
    # 保存的日志名称及模型训练结果路径
    if arg.rank == 0:
        os.makedirs('./results/', exist_ok=True)
        # 根据是否为测试模式决定是否创建 writer
        writer = SummaryWriter('runs/HiViT_Up_CosA') if not arg.test_only else None

    torch.manual_seed(arg.seed)
    np.random.seed(arg.seed)
    device = torch.device(f"cuda:{arg.gpu}")

    train_transform, val_transform = get_data_transforms(arg.img_size)
    
    full_train_dataset = load_data(os.path.join(arg.data_path, 'train'), train_transform)
    val_dataset_for_val_transform = load_data(os.path.join(arg.data_path, 'train'), val_transform)
    test_set = load_data(os.path.join(arg.data_path, 'test'), val_transform)

    dataset_size = len(full_train_dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)

    val_split = 0.2
    split_idx = int(np.floor(val_split * dataset_size))
    train_indices, val_indices = indices[split_idx:], indices[:split_idx]

    train_set = Subset(full_train_dataset, train_indices)
    val_set = Subset(val_dataset_for_val_transform, val_indices)

    train_sampler = DistributedSampler(train_set) if arg.distributed else None
    val_sampler = DistributedSampler(val_set, shuffle=False) if arg.distributed else None

    train_loader = DataLoader(train_set, batch_size=arg.batch_size, sampler=train_sampler, num_workers=arg.workers, pin_memory=True, shuffle=(train_sampler is None))
    val_loader = DataLoader(val_set, batch_size=arg.batch_size, sampler=val_sampler, num_workers=arg.workers, pin_memory=True)

    if arg.rank == 0:
        print(f"训练集: {len(train_set)}, 验证集: {len(val_set)}, 测试集: {len(test_set)}")

    # 创建模型
    model = HiViT_base(arg.classes).to(device)
    
    if arg.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[arg.gpu])

    best_epoch = 0

    if not arg.test_only:
        # 加载预训练权重
        if arg.pretrained_weights_url and not arg.resume:
            model_without_ddp = model.module if hasattr(model, 'module') else model
            load_pretrained_weights(model_without_ddp, arg.pretrained_weights_url, arg.classes)
        
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = AdamW(model.parameters(), lr=arg.lr, weight_decay=arg.weight_decay)
        
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-6 / arg.lr, total_iters=arg.warmup_epochs)
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=arg.epochs - arg.warmup_epochs, eta_min=arg.min_lr)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[arg.warmup_epochs])

        scaler = amp.GradScaler()

        dataset_name = os.path.basename(os.path.normpath(arg.data_path))
        best_ckpt_path = os.path.join('./results/', f'{dataset_name}_HiViT_best.pth')
        
        start_epoch = 1
        best_val_acc, epochs_no_improve = 0, 0

        if arg.resume and os.path.exists(arg.resume):
            print(f"=> 正在加载检查点: '{arg.resume}'")
            checkpoint = torch.load(arg.resume, map_location=device)
            model_to_load = model.module if arg.distributed else model
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model_to_load.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                best_val_acc = checkpoint['best_val_acc']
                best_epoch = checkpoint['epoch']
                epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
                print(f"=> 已加载检查点 '{arg.resume}' (从 epoch {start_epoch} 开始)")
            else:
                model_to_load.load_state_dict(checkpoint)
                print(f"=> 已从旧格式检查点 '{arg.resume}' 加载模型权重。")

        for epoch in range(start_epoch, arg.epochs + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, scaler, arg)
            val_acc = evaluate(model, val_loader, criterion, device, arg, desc=f"Epoch {epoch} Val")
            scheduler.step()

            if arg.rank == 0:
                print(f"[Epoch {epoch}] Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.2e}")
                
                if writer:
                    writer.add_scalar('Accuracy/train', train_acc, epoch)
                    writer.add_scalar('Accuracy/validation', val_acc, epoch)
                    writer.add_scalar('Loss/train', train_loss, epoch)
                    writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch
                    epochs_no_improve = 0
                    
                    model_to_save = model.module if arg.distributed else model
                    save_dict = {
                        'epoch': epoch,
                        'model_state_dict': model_to_save.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                        'best_val_acc': best_val_acc,
                        'epochs_no_improve': epochs_no_improve
                    }
                    torch.save(save_dict, best_ckpt_path)
                    print(f"*** 新最佳模型保存在 Epoch {best_epoch} (Val Acc: {best_val_acc:.2f}%) ***")
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= arg.patience:
                    print(f"连续 {arg.patience} 轮验证集性能未提升，触发早停。")
                    break
                
                if epoch > arg.overfit_check_epoch and (train_acc - val_acc > arg.overfit_gap_threshold):
                    print(f"训练/验证准确率差距 ({train_acc - val_acc:.2f}%) 超过阈值 {arg.overfit_gap_threshold}，触发早停。")
                    break

    if arg.distributed:
        dist.barrier()

    # --- 最终测试 ---
    # 在所有进程上确定要测试的 checkpoint 路径
    if arg.test_only:
        ckpt_path = arg.test_checkpoint
        if not ckpt_path:
            if arg.rank == 0:
                print("错误：仅测试模式需要通过 --test_checkpoint 指定模型路径。")
            if arg.distributed:
                dist.barrier()
            cleanup()
            sys.exit(1)
        if not ckpt_path.endswith(('.pth', '.pt')):
            if arg.rank == 0:
                print(f"错误：无效的 checkpoint 文件 '{ckpt_path}'。文件必须是 .pth 或 .pt 格式。")
            if arg.distributed:
                dist.barrier()
            cleanup()
            sys.exit(1)
        if arg.rank == 0:
            print(f"\n--- 仅测试模式, 使用模型: {ckpt_path} ---")
    else:
        dataset_name = os.path.basename(os.path.normpath(arg.data_path))
        ckpt_path = os.path.join('./results/', f'{dataset_name}_HiViT_best.pth')
        if arg.rank == 0:
            print(f"\n--- 训练结束, 使用 Epoch {best_epoch} 的最佳模型测试 ---")

    # 所有进程都参与测试
    if os.path.exists(ckpt_path):
        if arg.rank == 0:
            print(f"正在从 '{ckpt_path}' 加载最佳模型进行测试...")

        # 1. 加载权重到CPU, 使用 weights_only=True 避免安全风险
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=True)

        # 2. 创建模型并移动到各进程对应的 GPU
        test_model = HiViT_base(arg.classes).to(device)

        # 3. 如果是DDP，先用 DDP 包裹模型
        if arg.distributed:
            test_model = torch.nn.parallel.DistributedDataParallel(test_model, device_ids=[arg.gpu])

        # 4. 将权重加载到模型中
        model_to_load = test_model.module if arg.distributed else test_model
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model_to_load.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model_to_load.load_state_dict(checkpoint, strict=False)
        
        # 5. 创建测试数据加载器
        test_sampler = DistributedSampler(test_set, shuffle=False) if arg.distributed else None
        test_loader = DataLoader(test_set, batch_size=arg.batch_size, sampler=test_sampler, num_workers=arg.workers, pin_memory=True, shuffle=False)

        # 6. 执行评估
        test_acc = evaluate(test_model, test_loader, criterion=None, device=device, args=arg, desc="Testing")

        # 7. 只在 rank 0 打印结果
        if arg.rank == 0:
            print(f"最终测试 OA: {test_acc:.2f}%")
    else:
        if arg.rank == 0:
            print(f"未找到最佳模型文件 '{ckpt_path}'，跳过测试。")

    if arg.distributed:
        dist.barrier()
    
    if arg.rank == 0 and not arg.test_only and writer:
        writer.close()

    cleanup()

if __name__ == '__main__':
    # 将主逻辑移入 main 函数，便于组织和管理
    main()