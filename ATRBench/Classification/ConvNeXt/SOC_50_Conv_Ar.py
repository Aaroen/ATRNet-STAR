import sys
import torch
import numpy as np
import re
from tqdm.auto import tqdm
import argparse
import torch.nn as nn
import collections
import torchvision.transforms as transforms
import warnings
from utils.DataLoad import load_data
from utils.schedulers import create_scheduler
from model.ConvNeXt import convnext_1
import os
import subprocess
from datetime import datetime
from torch.utils.data import Subset, DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.optim.adamw import AdamW
from torch.cuda.amp import GradScaler, autocast
import pytz
import signal

warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*is deprecated", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.GradScaler.*is deprecated", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Grad strides do not match bucket view strides.*", category=UserWarning)

# --- 目录定义 ---
# 获取此文件所在的目录
_current_dir = os.path.dirname(os.path.abspath(__file__))
# 项目根目录是此文件目录的上三级
PROJECT_ROOT = os.path.abspath(os.path.join(_current_dir, '..', '..', '..'))
ddp_process = None

def parameter_setting():
    """参数设置函数"""
    parser = argparse.ArgumentParser(description='ConvNeXt Up')
    
    # --- 基本与路径参数 ---
    parser.add_argument('--model_name', default='ConvNeXt', type=str, help='模型名称，用于结果保存')
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
    parser.add_argument('--model_path', default='', type=str, help='模型检查点路径，用于恢复训练或测试')
    parser.add_argument('--resume', default=False, action='store_true', help='是否从检查点恢复训练 (默认为False, 需使用 --model_path 指定路径)')
    parser.add_argument('--test_only', default=False, action='store_true', help='仅运行测试模式 (默认为False, 需使用 --model_path 指定路径)')
    
    args = parser.parse_args()
    return args

def init_distributed_mode(args):
    """初始化 DDP 环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    else:
        return False

    args.distributed = True
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    dist.barrier()
    if args.rank == 0:
        print(f'DDP 环境初始化完成 (World Size: {args.world_size})', flush=True)
    return True

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

def get_data_transforms(img_size=224):
    """获取训练和验证/测试的数据变换"""
    # 训练集使用数据增强
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # 验证和测试集不使用数据增强
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform

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
        
        with autocast(dtype=torch.float16):
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
    
    loss_meter = torch.tensor(0.0, device=device)
    correct_meter = torch.tensor(0.0, device=device)
    total_samples_meter = torch.tensor(0.0, device=device)

    # 准备进度条
    progress_bar = tqdm(data_loader, desc=desc, disable=(args.rank != 0), dynamic_ncols=True)
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with autocast(dtype=torch.float16):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        loss_meter += loss.detach() * inputs.size(0)
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
        dist.barrier()
        final_loss = reduce_value(loss_meter, average=False)
        final_correct = reduce_value(correct_meter, average=False)
        final_samples = reduce_value(total_samples_meter, average=False)
    else:
        final_loss = loss_meter
        final_correct = correct_meter
        final_samples = total_samples_meter
        
    avg_loss = final_loss.item() / final_samples.item() if final_samples.item() > 0 else 0.0
    accuracy = 100. * final_correct.item() / final_samples.item() if final_samples.item() > 0 else 0.0
    return avg_loss, accuracy

def load_checkpoint_for_test(ckpt_path, num_classes, device, args):
    """为测试加载模型检查点。"""
    if args.rank == 0 and not os.path.exists(ckpt_path):
        print(f"错误：未找到检查点文件 '{ckpt_path}'，跳过测试。")
        return None

    # 在所有进程上同步，确保文件存在性判断一致
    if args.distributed:
        dist.barrier()
        
    if not os.path.exists(ckpt_path): # 对于非主进程，如果文件不存在则直接返回
        return None

    # 为避免内存尖峰，先将检查点加载到 CPU，并抑制 FutureWarning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        checkpoint = torch.load(ckpt_path, map_location='cpu')

    # 提取 state_dict，兼容旧格式
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    # 创建模型实例时，明确指定不使用预训练权重
    model = convnext_1(num_classes=num_classes, pretrained=False)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    
    # 如果是 DDP 模式，用 DDP 包裹模型
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    return model

def main(arg, device):
    # --- 参数校验 ---
    if arg.test_only and not arg.model_path:
        if arg.rank == 0:
            print("\n错误: 当使用 --test_only 时, 必须通过 --model_path 参数指定一个模型检查点路径。")
        if arg.distributed: dist.barrier()
        cleanup()
        sys.exit(1)
        
    if arg.resume and not arg.model_path:
        if arg.rank == 0:
            print("\n错误: 当使用 --resume 时, 必须通过 --model_path 参数指定一个检查点路径。")
        if arg.distributed: dist.barrier()
        cleanup()
        sys.exit(1)

    # --- 数据加载与变换 ---
    train_transform, val_transform = get_data_transforms(arg.img_size)
    test_set = load_data(os.path.join(arg.data_path, 'test'), val_transform)

    # --- 测试模式 ---
    if arg.test_only:
        if arg.rank == 0:
            print(f"--- 开始测试 | 模型: {os.path.basename(arg.model_path)} ---")
            print(f"测试集样本数: {len(test_set)}")

        test_model = load_checkpoint_for_test(arg.model_path, arg.classes, device, arg)
        if test_model:
            test_sampler = DistributedSampler(test_set, shuffle=False) if arg.distributed else None
            test_loader = DataLoader(test_set, batch_size=arg.batch_size, sampler=test_sampler, num_workers=arg.workers, pin_memory=True, shuffle=False)
            criterion = nn.CrossEntropyLoss()
            test_loss, test_acc = evaluate(test_model, test_loader, criterion, device, arg, desc="Testing")

            if arg.rank == 0:
                print(f"最终测试 -> Loss: {test_loss:.4f}, OA: {test_acc:.2f}%")
        
        if arg.distributed:
            dist.barrier()
        return  # 测试完成后直接返回

    # --- 训练模式 ---
    
    # 1. 定义日志和结果目录路径
    runs_root = os.path.join(PROJECT_ROOT, 'runs')
    results_root = os.path.join(PROJECT_ROOT, 'results')
    results_dir = os.path.join(results_root, arg.model_name)
    writer = None # 初始化 writer

    if arg.rank == 0:
        # 2. 仅在主进程创建目录和 TensorBoard writer
        os.makedirs(runs_root, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        tz = pytz.timezone('Asia/Shanghai')
        timestamp = datetime.now(tz).strftime('%Y%m%d_%H%M%S')
        log_dir = os.path.join(runs_root, f"{arg.model_name}_{arg.scheduler}_{timestamp}")
        writer = SummaryWriter(log_dir)
        print(f"TensorBoard 日志保存在: {log_dir}")

    full_train_dataset = load_data(os.path.join(arg.data_path, 'train'), train_transform)

    # 划分训练集和验证集
    val_split = 0.2
    dataset_size = len(full_train_dataset)
    indices = list(range(dataset_size))
    np.random.seed(arg.seed)
    np.random.shuffle(indices)
    split_idx = int(np.floor(val_split * dataset_size))
    train_indices, val_indices = indices[split_idx:], indices[:split_idx]
    
    train_set = Subset(full_train_dataset, train_indices)
    val_dataset_for_val_transform = load_data(os.path.join(arg.data_path, 'train'), val_transform)
    val_set = Subset(val_dataset_for_val_transform, val_indices)
    
    train_sampler = DistributedSampler(train_set) if arg.distributed else None
    val_sampler = DistributedSampler(val_set, shuffle=False) if arg.distributed else None

    train_loader = DataLoader(train_set, batch_size=arg.batch_size, sampler=train_sampler, num_workers=arg.workers, pin_memory=True, shuffle=(train_sampler is None))
    val_loader = DataLoader(val_set, batch_size=arg.batch_size, sampler=val_sampler, num_workers=arg.workers, pin_memory=True, shuffle=False)
    
    if arg.rank == 0:
        print(f"训练集: {len(train_set)}, 验证集: {len(val_set)}, 测试集: {len(test_set)}")

    # --- 训练核心对象初始化 ---
    
    use_online_pretrain = not arg.no_pretrain and not arg.pretrain_path
    model = convnext_1(num_classes=arg.classes, pretrained=use_online_pretrain).to(device)

    if not arg.no_pretrain and arg.pretrain_path:
        if os.path.exists(arg.pretrain_path):
            if arg.rank == 0:
                print(f"=> 正在加载本地预训练权重: '{arg.pretrain_path}'")
            
            checkpoint = torch.load(arg.pretrain_path, map_location=device)
            state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint.get('model_state_dict', checkpoint)))
            model.load_state_dict(state_dict, strict=False)
            
            if arg.rank == 0:
                print(f"=> 本地预训练权重加载完成。")
        else:
            if arg.rank == 0:
                print(f"警告: 提供的预训练路径 '{arg.pretrain_path}' 不存在。将跳过加载本地预训练权重。")

    if arg.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[arg.gpu])
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=arg.lr, weight_decay=arg.weight_decay)
    scheduler = create_scheduler(optimizer, arg)
    scaler = GradScaler()
    
    start_epoch = 1
    best_val_acc = 0.0
    epochs_no_improve = 0
    
    tmp_best_ckpt_path = os.path.join(results_dir, f"_best_tmp_{arg.model_name}.pth")

    if arg.resume and arg.model_path and os.path.exists(arg.model_path):
        if arg.rank == 0:
            print(f"=> 正在加载检查点: '{arg.model_path}'")
        checkpoint = torch.load(arg.model_path, map_location=device)
        
        model_to_load = model.module if arg.distributed else model
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model_to_load.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_acc = checkpoint['best_val_acc']
            epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
            if arg.rank == 0:
                print(f"=> 已加载检查点 '{arg.model_path}' (从 epoch {start_epoch} 开始)")
        else:
            model_to_load.load_state_dict(checkpoint)
            if arg.rank == 0:
                print(f"=> 已从旧格式检查点 '{arg.model_path}' 加载模型权重。优化器和调度器将重新开始。")

    # --- 训练循环 ---
    for epoch in range(start_epoch, arg.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, scaler, arg)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, arg, desc=f"Epoch {epoch} Val")
        scheduler.step()

        if arg.rank == 0:
            print(f"[Epoch {epoch}] Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Val Loss: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
            if writer:
                writer.add_scalar('Accuracy/train', train_acc, epoch)
                writer.add_scalar('Accuracy/validation', val_acc, epoch)
                writer.add_scalar('Loss/train', train_loss, epoch)
                writer.add_scalar('Loss/validation', val_loss, epoch)
                writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0
                
                model_to_save = model.module if arg.distributed else model
                save_dict = {
                    'epoch': epoch,
                    'model_state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'best_val_acc': best_val_acc,
                }
                torch.save(save_dict, tmp_best_ckpt_path)
                print(f"*** 新最佳模型(临时)保存在 Epoch {epoch} (Val Acc: {best_val_acc:.2f}%) ***")
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= arg.patience:
                print(f"连续 {arg.patience} 轮验证集性能未提升，触发早停。")
                break
            
            if epoch > arg.overfit_check_epoch and (train_acc - val_acc > arg.overfit_gap_threshold):
                print(f"训练/验证准确率差距 ({train_acc - val_acc:.2f}%) 超过阈值 {arg.overfit_gap_threshold}，触发早停。")
                break

    best_ckpt_path = tmp_best_ckpt_path if os.path.exists(tmp_best_ckpt_path) else ""

    if arg.distributed:
        dist.barrier()

    # --- 训练后最终测试 ---
    if not best_ckpt_path or not os.path.exists(best_ckpt_path):
        if arg.rank == 0:
            print("\n未找到可供测试的最佳模型，跳过最终测试。")
    else:
        if arg.rank == 0:
            print(f"\n--- 开始测试 | 模型: {os.path.basename(best_ckpt_path)} ---")
        
        test_model = load_checkpoint_for_test(best_ckpt_path, arg.classes, device, arg)
        if test_model:
            test_sampler = DistributedSampler(test_set, shuffle=False) if arg.distributed else None
            test_loader = DataLoader(test_set, batch_size=arg.batch_size, sampler=test_sampler, num_workers=arg.workers, pin_memory=True, shuffle=False)
            test_loss, test_acc = evaluate(test_model, test_loader, criterion, device, arg, desc="Testing")

            if arg.rank == 0:
                print(f"最终测试 -> Loss: {test_loss:.4f}, OA: {test_acc:.2f}%")
                
                perf_val_str = f"{test_acc:.3f}".replace('.', '')
                final_name = f"{arg.model_name}_{arg.scheduler}_{perf_val_str}.pth"
                
                model_dir = os.path.dirname(best_ckpt_path)
                final_path = os.path.join(model_dir, final_name)
                
                os.rename(best_ckpt_path, final_path)
                print(f"模型已重命名为: {final_name}")

    if arg.distributed:
        dist.barrier()

    if arg.rank == 0 and writer is not None:
        writer.close()

    cleanup()

if __name__ == '__main__':
    # --- 自引导 DDP ---
    # 检查是否在 DDP 环境中
    if 'RANK' not in os.environ and 'WORLD_SIZE' not in os.environ and torch.cuda.device_count() > 1:
        
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
        n_gpus = torch.cuda.device_count()
        
        print(f"--- 检测到非DDP环境, 将使用 {n_gpus} 个GPU以DDP模式重新启动 ---")
        
        os.environ.update({'MASTER_ADDR': 'localhost', 'MASTER_PORT': '12355'})
        cmd = ["torchrun", "--standalone", "--nnodes=1", f"--nproc_per_node={n_gpus}", sys.argv[0]] + sys.argv[1:]
        
        try:
            ddp_process = subprocess.Popen(cmd)
            ddp_process.wait()
        except KeyboardInterrupt:
            pass # 信号处理器会处理
        except Exception as e:
            # 只有在主进程中才打印错误
            if 'RANK' not in os.environ:
                print(f"\nDDP 启动器发生错误: {e}")
        finally:
            # 确保在任何情况下，如果子进程还在运行，都尝试终止它
            if ddp_process and ddp_process.poll() is None:
                ddp_process.terminate()
            if 'RANK' not in os.environ:
                print("\nDDP 父进程执行完毕。")
            sys.exit(0)

    # --- 主程序开始 (已在DDP环境中) ---
    arg = parameter_setting()
    
    # 初始化DDP，此时应该总能成功
    init_distributed_mode(arg)

    torch.manual_seed(arg.seed)
    np.random.seed(arg.seed)
    device = torch.device(f"cuda:{arg.gpu}")

    main(arg, device)