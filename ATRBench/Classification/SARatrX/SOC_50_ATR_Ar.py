import sys
import torch
import numpy as np
import re
from tqdm.auto import tqdm
import argparse
import torch.nn as nn
import torchvision.transforms as transforms
import warnings
from utils.DataLoad import load_data
from model.HiVit import HiViT_base
import os
import subprocess
from datetime import datetime
import pytz
from torch.utils.data import Subset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.optim.adamw import AdamW
from torch.cuda import amp
import torch.backends.cudnn as cudnn
import signal

# 屏蔽特定的 FutureWarning 和 UserWarning
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*is deprecated", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.GradScaler.*is deprecated", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Grad strides do not match bucket view strides.*", category=UserWarning)

# --- 目录定义 ---
_current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_current_dir, '..', '..', '..'))
ddp_process = None

# --- DDP 辅助函数 ---
def is_main_process():
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0

def get_rank():
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def get_world_size():
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

def parameter_setting():
    """参数设置函数"""
    parser = argparse.ArgumentParser(description='SARatrX (HiViT)_Up')
    
    # --- 基本与路径参数 ---
    parser.add_argument('--model_name', default='SARatrX_HiViT', type=str, help='模型名称，用于结果保存')
    parser.add_argument('--data_path', default='../../datasets/SOC_50classes/', type=str, help='数据集路径')
    parser.add_argument('--epochs', default=256, type=int, help='训练总轮数')
    parser.add_argument('--classes', default=50, type=int, help='类别数量')
    parser.add_argument('--batch_size', default=128, type=int, help='单个GPU的批处理大小')
    parser.add_argument('--workers', default=8, type=int, help='数据加载的工作线程数')
    parser.add_argument('--seed', default=42, type=int, help='随机种子')
    parser.add_argument('--img_size', default=224, type=int, help='输入图像尺寸')
    
    # --- 学习率与优化器参数 ---
    parser.add_argument('--scheduler', default='warmup_cosine', type=str, choices=['warmup_cosine', 'cosine', 'step'], help='学习率调度器类型')
    parser.add_argument('--lr', default=5e-4, type=float, help='学习率')
    parser.add_argument('--warmup_epochs', default=20, type=int, help='预热轮数 (仅用于 warmup_cosine)')
    parser.add_argument('--min_lr', default=1e-6, type=float, help='最低学习率 (仅用于余弦类调度器)')
    parser.add_argument('--weight_decay', default=0.05, type=float, help='权重衰减')
    parser.add_argument('--lr_step_size', default=30, type=int, help='StepLR的步长')
    parser.add_argument('--lr_gamma', default=0.1, type=float, help='StepLR的衰减率')
    
    # --- 正则化与早停参数 ---
    parser.add_argument('--patience', default=30, type=int, help='早停: 验证集性能无提升的等待轮数')
    parser.add_argument('--overfit_gap_threshold', default=20.0, type=float, help='早停: 训练与验证准确率差距阈值')
    parser.add_argument('--overfit_check_epoch', default=20, type=int, help='开始检查过拟合的轮数')
    
    # --- 训练稳定性参数 ---
    parser.add_argument('--clip_grad', default=1.0, type=float, help='梯度裁剪阈值 (<=0 表示不裁剪)')

    # --- 预训练控制 ---
    parser.add_argument('--no_pretrain', default=False, action='store_true', help='禁用模型内部硬编码的预训练权重 (默认为False, 即使用预训练)')
    parser.add_argument('--pretrain_path', default='', type=str, help='本地预训练权重文件路径 (如果提供, 将覆盖在线下载)')

    # --- DDP 相关参数 ---
    parser.add_argument('--dist_url', default='env://', help='DDP 使用的 URL')
    
    # --- 恢复与测试 ---
    parser.add_argument('--model_path', default='results/SARatrX_HiViT/SOC_50classes_SARatrX_SGDR_95074.pth', type=str, help='模型检查点路径，用于恢复训练或测试')
    parser.add_argument('--resume', default=False, action='store_true', help='是否从检查点恢复训练 (默认为False, 需使用 --model_path 指定路径)')
    parser.add_argument('--test_only', default=True, action='store_true', help='仅运行测试模式 (默认为False, 需使用 --model_path 指定路径)')
    
    args = parser.parse_args()
    return args

def create_scheduler(optimizer, args):
    """根据参数创建学习率调度器"""
    if args.scheduler == 'warmup_cosine':
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-6 / args.lr, total_iters=args.warmup_epochs
        )
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=args.min_lr
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[args.warmup_epochs]
        )
    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    elif args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    else:
        raise ValueError(f"不支持的调度器类型: {args.scheduler}")
    return scheduler

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

def train_one_epoch(model, data_loader, optimizer, criterion, device, epoch, scaler, args):
    """单轮训练循环，包含 DDP 同步和 AMP"""
    model.train()
    if args.distributed:
        data_loader.sampler.set_epoch(epoch)
    loss_meter, correct_meter = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
    
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch}/{args.epochs}", disable=(not is_main_process()), dynamic_ncols=True)

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
        dist.barrier()
        total_loss, total_correct = reduce_value(loss_meter, False), reduce_value(correct_meter, False)
    else:
        total_loss, total_correct = loss_meter, correct_meter
    
    avg_loss = total_loss.item() / len(data_loader.dataset)
    accuracy = 100. * total_correct.item() / len(data_loader.dataset)
    return avg_loss, accuracy

@torch.no_grad()
def evaluate(model, data_loader, criterion, device, args, desc=""):
    """评估函数，用于验证集和测试集，并实时显示精度"""
    model.eval()
    
    loss_meter = torch.tensor(0.0, device=device)
    correct_meter = torch.tensor(0.0, device=device)
    total_samples_meter = torch.tensor(0.0, device=device)

    progress_bar = tqdm(data_loader, desc=desc, disable=(not is_main_process()), dynamic_ncols=True)
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        loss_meter += loss.detach() * inputs.size(0)
        correct_meter += (outputs.argmax(1) == labels).sum()
        total_samples_meter += inputs.size(0)

        # 在 DDP 环境下，同步各个进程的中间结果以计算实时准确率
        if args.distributed:
            synced_correct = reduce_value(correct_meter.clone(), average=False)
            synced_total = reduce_value(total_samples_meter.clone(), average=False)
        else:
            synced_correct = correct_meter
            synced_total = total_samples_meter
        
        if is_main_process():
            current_acc = 100. * synced_correct.item() / synced_total.item() if synced_total.item() > 0 else 0.0
            progress_bar.set_description(f"{desc} (OA: {current_acc:.2f}%)")
    
    # --- DDP 同步最终结果 ---
    if args.distributed:
        dist.barrier()
        total_loss = reduce_value(loss_meter, False)
        total_correct = reduce_value(correct_meter, False)
        total_samples = reduce_value(total_samples_meter, False)
    else:
        total_loss = loss_meter
        total_correct = correct_meter
        total_samples = total_samples_meter
        
    avg_loss = total_loss.item() / total_samples.item() if total_samples.item() > 0 else 0.0
    accuracy = 100. * total_correct.item() / total_samples.item() if total_samples.item() > 0 else 0.0
    return avg_loss, accuracy

def init_distributed_mode(args):
    """初始化 DDP 环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    else:
        args.rank = 0
        args.world_size = 1
        args.gpu = 0
        return False

    args.distributed = True
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    dist.barrier()
    if is_main_process():
        print(f'DDP 环境初始化完成 (World Size: {args.world_size})', flush=True)
    return True

def cleanup():
    """清理 DDP 环境"""
    if dist.is_initialized():
        dist.destroy_process_group()

def reduce_value(value, average=True):
    """在所有 DDP 进程中同步和平均一个值"""
    world_size = get_world_size()
    if world_size < 2:
        return value
    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value /= world_size
        return value

def main(arg, device):
    if (arg.test_only or arg.resume) and not arg.model_path:
        if is_main_process(): print(f"错误: --{'test_only' if arg.test_only else 'resume'} 必须通过 --model_path 提供模型文件。")
        sys.exit(1)
    seed = arg.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    
    # --- 数据加载 ---
    train_transform, test_transform = get_data_transforms(arg.img_size)
    test_dataset = load_data(os.path.join(arg.data_path, 'test'), test_transform)

    # --- 模型创建与加载 ---
    use_pretrain = not arg.no_pretrain
    pretrain_path_to_load = arg.pretrain_path if use_pretrain and not arg.resume and not arg.test_only else None

    if use_pretrain and pretrain_path_to_load and not os.path.exists(pretrain_path_to_load):
        if is_main_process():
            print(f"--- 警告: 预训练权重文件 {pretrain_path_to_load} 未找到, 将随机初始化模型 ---")
        pretrain_path_to_load = None

    if is_main_process() and pretrain_path_to_load:
        print(f"--- 加载本地预训练模型: {pretrain_path_to_load} ---")

    model = HiViT_base(num_classes=arg.classes, pretrained=(pretrain_path_to_load is not None), pretrain_path=pretrain_path_to_load)

    if arg.model_path and os.path.exists(arg.model_path):
        if is_main_process():
            print(f"--- 加载检查点 | 模型: {os.path.basename(arg.model_path)} ---")
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            checkpoint = torch.load(arg.model_path, map_location='cpu')
            
        state_dict = checkpoint.get('model', checkpoint.get('model_state_dict', checkpoint))
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        load_result = model.load_state_dict(new_state_dict, strict=False)
        
        if is_main_process():
            if load_result.missing_keys or load_result.unexpected_keys:
                print("--- 警告: 模型结构与检查点不完全匹配 ---")
                if load_result.missing_keys: print(f"  - {len(load_result.missing_keys)}个权重丢失: {load_result.missing_keys[:2]}")
                if load_result.unexpected_keys: print(f"  - {len(load_result.unexpected_keys)}个权重多余: {load_result.unexpected_keys[:2]}")

    model.to(device)

    model_without_ddp = model
    if arg.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[arg.gpu], find_unused_parameters=False)
        model_without_ddp = model.module
    
    # --- 仅测试模式 ---
    if arg.test_only:
        test_sampler = DistributedSampler(test_dataset, shuffle=False) if arg.distributed else SequentialSampler(test_dataset)
        test_loader = DataLoader(test_dataset, batch_size=arg.batch_size, sampler=test_sampler, num_workers=arg.workers, pin_memory=True)
        # 需先定义 criterion
        criterion = nn.CrossEntropyLoss()
        test_loss, test_acc = evaluate(model, test_loader, criterion, device, arg, "Testing")
        if is_main_process():
            print(f"最终测试 -> Loss: {test_loss:.4f}, OA: {test_acc:.2f}%")
        return

    # --- 训练设置 ---
    full_train_dataset = load_data(os.path.join(arg.data_path, 'train'), train_transform)
    val_split = 0.2
    dataset_size = len(full_train_dataset)
    indices = list(range(dataset_size))
    np.random.seed(arg.seed)
    np.random.shuffle(indices)
    split_idx = int(np.floor(val_split * dataset_size))
    train_indices, val_indices = indices[split_idx:], indices[:split_idx]
    
    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset_with_val_transform = load_data(os.path.join(arg.data_path, 'train'), test_transform)
    val_dataset = Subset(val_dataset_with_val_transform, val_indices)

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if arg.distributed else RandomSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if arg.distributed else SequentialSampler(val_dataset)
    
    train_loader = DataLoader(train_dataset, batch_size=arg.batch_size, sampler=train_sampler, num_workers=arg.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=arg.batch_size, sampler=val_sampler, num_workers=arg.workers, pin_memory=True)
    
    if is_main_process(): print(f"训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}, 测试集: {len(test_dataset)}")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model_without_ddp.parameters(), lr=arg.lr, weight_decay=arg.weight_decay)
    scheduler = create_scheduler(optimizer, arg)
    scaler = amp.GradScaler()
    writer = None
    if is_main_process():
        timestamp = datetime.now(pytz.timezone('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
        log_dir = os.path.join(PROJECT_ROOT, 'runs', f"{arg.model_name}_{arg.scheduler}_{timestamp}")
        writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard 日志保存在: {log_dir}")
    
    # --- 训练循环 ---
    start_epoch = 0
    best_val_acc = 0.0
    epochs_no_improve = 0
    
    if arg.resume:
        if 'epoch' in checkpoint: start_epoch = checkpoint['epoch'] + 1
        if 'optimizer' in checkpoint: optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint: scheduler.load_state_dict(checkpoint['scheduler'])
        if 'best_val_acc' in checkpoint: best_val_acc = checkpoint['best_val_acc']
        if is_main_process(): print(f"从 epoch {start_epoch} 继续训练.")

    save_path = os.path.join(PROJECT_ROOT, 'results', arg.model_name)
    if is_main_process():
        os.makedirs(save_path, exist_ok=True)
    tmp_best_ckpt_path = os.path.join(save_path, f"_best_tmp_{arg.model_name}.pth")

    for epoch in range(start_epoch, arg.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, scaler, arg)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, arg, "Validating")
        scheduler.step()

        if is_main_process():
            print(f"Epoch {epoch+1}/{arg.epochs} -> Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            if writer:
                writer.add_scalar('Loss/train', train_loss, epoch)
                writer.add_scalar('Loss/validation', val_loss, epoch)
                writer.add_scalar('Accuracy/train', train_acc, epoch)
                writer.add_scalar('Accuracy/validation', val_acc, epoch)
                writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0
                
                checkpoint_to_save = {
                    'model': model_without_ddp.state_dict(), 'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(), 'epoch': epoch, 'args': arg, 'best_val_acc': best_val_acc
                }
                torch.save(checkpoint_to_save, tmp_best_ckpt_path)
                print(f"*** 新最佳模型(临时)已保存: {os.path.basename(tmp_best_ckpt_path)} (Val Acc: {best_val_acc:.2f}%) ***")
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= arg.patience:
                print(f"验证集性能在 {arg.patience} 轮内无提升，触发早停。")
                break
            if epoch >= arg.overfit_check_epoch and (train_acc - val_acc) > arg.overfit_gap_threshold:
                print(f"训练集与验证集准确率差距 ({train_acc - val_acc:.2f}%) 超过阈值 {arg.overfit_gap_threshold}%，触发早停。")
                break
        
        if arg.distributed:
            dist.barrier()

    # --- 最终测试 ---
    best_ckpt_path = tmp_best_ckpt_path if os.path.exists(tmp_best_ckpt_path) else ""
    if not best_ckpt_path:
        if is_main_process(): print("\n未找到可供测试的最佳模型，跳过最终测试。")
    else:
        if is_main_process():
            print(f"\n--- 开始测试 | 模型: {os.path.basename(best_ckpt_path)} ---")
        
        checkpoint = torch.load(best_ckpt_path, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])

        test_sampler = DistributedSampler(test_dataset, shuffle=False) if arg.distributed else SequentialSampler(test_dataset)
        test_loader = DataLoader(test_dataset, batch_size=arg.batch_size, sampler=test_sampler, num_workers=arg.workers, pin_memory=True)
        
        test_loss, test_acc = evaluate(model, test_loader, criterion, device, arg, "Testing")
        
        if is_main_process():
            print(f"最终测试 -> Loss: {test_loss:.4f}, OA: {test_acc:.2f}%")
            
            clean_scheduler_name = re.sub(r'[^a-zA-Z0-9_-]', '', arg.scheduler)
            final_name = f"SOC_{arg.classes}classes_{arg.model_name}_{clean_scheduler_name}_{int(test_acc * 1000)}.pth"
            final_path = os.path.join(save_path, final_name)
            
            os.rename(best_ckpt_path, final_path)
            print(f"模型已重命名为: {final_name}")

    if writer and is_main_process():
        writer.close()
    
    if arg.distributed:
        cleanup()

if __name__ == '__main__':
    if "RANK" not in os.environ and torch.cuda.device_count() > 1:
        
        def signal_handler(sig, frame):
            """优雅地处理中断信号，终止子进程。"""
            global ddp_process
            if ddp_process:
                print("\n捕获到中断信号，正在终止 DDP 子进程...")
                ddp_process.terminate()
                try:
                    ddp_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print("子进程未在5秒内响应，强制终止...")
                    ddp_process.kill()
            print("启动器进程退出。")
            sys.exit(0)

        # 注册信号处理器
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        os.environ['OMP_NUM_THREADS'] = '1'
        print(f"--- 检测到非DDP环境, 将使用 {torch.cuda.device_count()} 个GPU以DDP模式重新启动 ---")
        os.environ.update({'MASTER_ADDR': 'localhost', 'MASTER_PORT': '12355'})
        cmd = ["torchrun", "--standalone", "--nnodes=1", f"--nproc_per_node={torch.cuda.device_count()}", sys.argv[0]] + sys.argv[1:]
        
        try:
            ddp_process = subprocess.Popen(cmd)
            ddp_process.wait()
        except KeyboardInterrupt:
            pass # 信号处理器会处理
        except Exception as e:
            if is_main_process(): print(f"\nDDP 启动器发生错误: {e}")
        finally:
            if ddp_process and ddp_process.poll() is None:
                ddp_process.terminate()
            if is_main_process(): print("\nDDP 父进程执行完毕。")
            sys.exit(0)
    
    arg = parameter_setting()
    init_distributed_mode(arg)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        main(arg, device)
    finally:
        cleanup()
