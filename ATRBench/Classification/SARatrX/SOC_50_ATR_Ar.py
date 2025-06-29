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
import shutil
from torch.utils.data import Subset, DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from tqdm.auto import tqdm
from torch.optim.adamw import AdamW
import itertools
from torch.cuda import amp

# DDP and AMP utility functions from previous scripts

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

def parameter_setting():
    """参数设置函数"""
    parser = argparse.ArgumentParser(description='SARatrX (HiViT)_Up')
    
    # --- 基本参数 ---
    parser.add_argument('--data_path', type=str, default='../../datasets/SOC_50classes/', help='数据集路径')
    parser.add_argument('--epochs', type=int, default=365, help='训练总轮数')
    parser.add_argument('--classes', type=int, default=50, help='类别数量')
    parser.add_argument('--batch_size', type=int, default=128, help='单个GPU的批处理大小')
    parser.add_argument('--workers', type=int, default=8, help='数据加载的工作线程数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--img_size', type=int, default=224, help='输入图像尺寸')
    
    # --- 学习率与优化器参数 ---
    parser.add_argument('--lr', type=float, default=5e-4, help='学习率')
    parser.add_argument('--warmup_epochs', type=int, default=20, help='预热轮数')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='最低学习率')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='权重衰减')
    parser.add_argument('--T_0', type=int, default=15, help='余弦退火调度器的初始周期长度 (epochs)')
    parser.add_argument('--T_mult', type=int, default=2, help='余弦退火周期增长的乘数')
    
    # --- 正则化与早停参数 ---
    parser.add_argument('--patience_cycles', type=int, default=2, help='早停: 验证集性能无提升的等待学习率周期数')
    parser.add_argument('--overfit_gap_threshold', type=float, default=20.0, help='早停: 训练与验证准确率差距阈值')
    parser.add_argument('--overfit_check_epoch', type=int, default=20, help='开始检查过拟合的轮数')
    
    # --- 训练稳定性参数 ---
    parser.add_argument('--clip_grad', type=float, default=1.0, help='梯度裁剪阈值 (<=0 表示不裁剪)')

    # --- 预训练与断点续训 ---
    parser.add_argument('--pretrained_weights', type=str, default='ATRBench/Classification/SARatrX/model/pretrained/mae_hivit_base_1600ep.pth', help='预训练权重文件路径 (可选)')
    parser.add_argument('--resume', type=str, default='', help='断点续训的 checkpoint 路径')

    # --- DDP 相关参数 ---
    parser.add_argument('--dist_url', default='env://', help='DDP 使用的 URL')
    
    args = parser.parse_args()
    return args

def get_data_transforms(img_size=224):
    """获取包含强力数据增强的变换"""
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
    ])
    return train_transform, val_transform

def load_pretrained_weights(model, weights_path, num_classes):
    """加载预训练权重，并智能处理分类头不匹配的问题"""
    if os.path.isfile(weights_path):
        try:
            print(f"正在从 {weights_path} 加载预训练权重...")
            checkpoint = torch.load(weights_path, map_location='cpu', weights_only=True)

            # 兼容不同保存方式的 checkpoint
            if 'model' in checkpoint:
                checkpoint_model = checkpoint['model']
            else:
                checkpoint_model = checkpoint

            # 移除 'module.' 前缀 (DDP 保存时会自动添加)
            state_dict = {k.replace('module.', ''): v for k, v in checkpoint_model.items()}

            # 检查并处理分类头尺寸不匹配的问题
            for k in ['head.weight', 'head.bias']:
                if k in state_dict and state_dict[k].shape != model.head.state_dict()[k.replace('head.','')].shape:
                    print(f"移除尺寸不匹配的键: {k}")
                    del state_dict[k]
            
            # 加载权重
            msg = model.load_state_dict(state_dict, strict=False)
            print("预训练权重加载完毕。")
            if msg.missing_keys:
                print("丢失的键:", msg.missing_keys)
            if msg.unexpected_keys:
                print("非预期的键:", msg.unexpected_keys)

        except Exception as e:
            print(f"加载预训练权重失败: {e}。将从零开始训练。")
    else:
        print("未提供预训练权重文件或文件不存在，将从零开始训练。")

def train_one_epoch(model, data_loader, optimizer, criterion, device, epoch, scaler, args):
    """单轮训练循环，包含 DDP 同步和 AMP"""
    model.train()
    sampler = data_loader.sampler
    if args.distributed and sampler is not None:
        sampler.set_epoch(epoch)

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

        loss_meter += loss.detach()
        correct_meter += (outputs.argmax(1) == labels).sum()

    if args.distributed:
        total_loss = reduce_value(loss_meter, average=False)
        total_correct = reduce_value(correct_meter, average=False)
        dataset_len = len(sampler.dataset)
    else:
        total_loss = loss_meter
        total_correct = correct_meter
        dataset_len = len(data_loader.dataset)
    
    avg_loss = total_loss.item() / dataset_len
    accuracy = 100. * total_correct.item() / dataset_len
    return avg_loss, accuracy

@torch.no_grad()
def evaluate(model, data_loader, criterion, device, args):
    """评估函数，用于验证集和测试集"""
    model.eval()
    correct_meter = torch.tensor(0.0, device=device)

    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with amp.autocast():
            outputs = model(inputs)
        correct_meter += (outputs.argmax(1) == labels).sum()
    
    if args.distributed:
        total_correct = reduce_value(correct_meter, average=False)
        total_samples = len(data_loader.sampler.dataset)
    else:
        total_correct = correct_meter
        total_samples = len(data_loader.dataset)
        
    accuracy = 100. * total_correct.item() / total_samples
    return accuracy

if __name__ == '__main__':
    arg = parameter_setting()
    init_distributed_mode(arg)
    
    if arg.rank == 0:
        os.makedirs('./results/', exist_ok=True)
        writer = SummaryWriter('runs/SARatrX_Exp_CoaW')

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

    model = HiViT_base(num_classes=arg.classes, rpe=False).to(device)
    
    # 加载预训练权重
    if arg.pretrained_weights and not arg.resume: # 只有在不恢复训练时才加载预训练权重
        model_without_ddp = model.module if hasattr(model, 'module') else model
        load_pretrained_weights(model_without_ddp, arg.pretrained_weights, arg.classes)

    if arg.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[arg.gpu])

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=arg.lr, weight_decay=arg.weight_decay)
    
    #  "预热 + 余弦退火" 学习率调度器 ---
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-6 / arg.lr, total_iters=arg.warmup_epochs)
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=arg.T_0, T_mult=arg.T_mult, eta_min=arg.min_lr)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[arg.warmup_epochs])

    scaler = amp.GradScaler()

    dataset_name = os.path.basename(os.path.normpath(arg.data_path))
    best_ckpt_path = os.path.join('./results/', f'{dataset_name}_SARatrX_best.pth')
    
    start_epoch = 1
    best_val_acc, best_epoch = 0, 0
    cycles_no_improve = 0
    best_acc_at_last_cycle_check = 0.0
    
    if arg.resume and os.path.isfile(arg.resume):
        print(f"=> 从 '{arg.resume}' 加载 checkpoint")
        checkpoint = torch.load(arg.resume, map_location=device)
        
        model_to_load = model.module if arg.distributed else model
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        best_epoch = checkpoint.get('epoch', 0)
        cycles_no_improve = checkpoint.get('cycles_no_improve', 0)
        best_acc_at_last_cycle_check = checkpoint.get('best_acc_at_last_cycle_check', 0.0)
        print(f"=> 从 epoch {start_epoch} 继续训练")
    
    for epoch in range(start_epoch, arg.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, scaler, arg)
        val_acc = evaluate(model, val_loader, criterion, device, arg)
        scheduler.step()

        if arg.rank == 0:
            print(f"[Epoch {epoch}] Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.2e}")
            
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/validation', val_acc, epoch)
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
            
            # --- 检查早停条件 ---
            # 1. 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                model_to_save = model.module if arg.distributed else model
                state_to_save = {
                    'epoch': epoch,
                    'model_state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'best_val_acc': best_val_acc,
                    'cycles_no_improve': cycles_no_improve,
                    'best_acc_at_last_cycle_check': best_acc_at_last_cycle_check
                }
                torch.save(state_to_save, best_ckpt_path)
                print(f"*** 新最佳模型保存在 Epoch {best_epoch} (Val Acc: {best_val_acc:.2f}%) ***")

            # 2. 基于学习率周期的早停
            if epoch > arg.warmup_epochs:
                current_lr = optimizer.param_groups[0]['lr']
                if abs(current_lr - arg.min_lr) < 1e-8:
                    print(f"--- Cycle ended at epoch {epoch}. Current best acc: {best_val_acc:.2f}%. Best at last check: {best_acc_at_last_cycle_check:.2f}% ---")
                    if best_val_acc > best_acc_at_last_cycle_check:
                        cycles_no_improve = 0
                        best_acc_at_last_cycle_check = best_val_acc
                        print("--- Accuracy improved. Resetting early stopping counter. ---")
                    else:
                        cycles_no_improve += 1
                        print(f"--- No accuracy improvement. Counter: {cycles_no_improve}/{arg.patience_cycles}. ---")
                    
                    if cycles_no_improve >= arg.patience_cycles:
                        print(f"Validation accuracy has not improved for {arg.patience_cycles} cycles. Triggering early stopping.")
                        break

            # 3. 基于过拟合差距的早停
            if epoch > arg.overfit_check_epoch and (train_acc - val_acc > arg.overfit_gap_threshold):
                print(f"训练/验证准确率差距 ({train_acc - val_acc:.2f}%) 超过阈值 {arg.overfit_gap_threshold}，触发早停。")
                break
    
    if arg.distributed:
        dist.barrier()

    if arg.rank == 0:
        print(f"\n--- 训练结束, 使用 Epoch {best_epoch} 的最佳模型测试 ---")
        if os.path.exists(best_ckpt_path):
            print(f"正在从 '{best_ckpt_path}' 加载最佳模型进行测试...")
            checkpoint = torch.load(best_ckpt_path, map_location=device)
            test_model = HiViT_base(num_classes=arg.classes, rpe=False).to(device)
            test_model.load_state_dict(checkpoint['model_state_dict'])
            
            test_loader_single = DataLoader(test_set, batch_size=arg.batch_size, shuffle=False, num_workers=arg.workers)
            test_acc = model_test(test_model, test_loader_single)
            print(f"最终测试 OA: {test_acc:.2f}%")
            
            # 保存结果
            results_path = os.path.join('./results/', f'{dataset_name}_SARatrX_results.npy')
            np.save(results_path, np.array([test_acc]))
            print(f"测试结果已保存到: {results_path}")
        else:
            print("未找到最佳模型文件，跳过测试。")
    
    if arg.distributed:
        dist.barrier()

    if arg.rank == 0:
        writer.close()

    cleanup()
