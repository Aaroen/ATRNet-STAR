# -*- coding: utf-8 -*-
import sys
import torch
import numpy as np
import re
from tqdm.auto import tqdm
import argparse
import torch.nn as nn
import collections
import torchvision.transforms as transforms
from utils.DataLoad import load_data
from utils.TrainTest import model_train, model_val, model_test
from model.ConvNeXt import convnext_1
import os
from torch.utils.data import Subset, DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.optim.adamw import AdamW

# 脚本根目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

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
    parser = argparse.ArgumentParser(description='ConvNeXt Up')
    
    # --- 基本参数 ---
    parser.add_argument('--data_path', type=str, default='../../datasets/SOC_50classes/', help='数据集路径')
    parser.add_argument('--epochs', type=int, default=200, help='训练总轮数')
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
    
    # --- 正则化与早停参数 ---
    parser.add_argument('--patience', type=int, default=30, help='早停: 验证集性能无提升的等待轮数')
    parser.add_argument('--overfit_gap_threshold', type=float, default=20.0, help='早停: 训练与验证准确率差距阈值')
    
    # --- 训练稳定性参数 ---
    parser.add_argument('--clip_grad', type=float, default=1.0, help='梯度裁剪阈值 (<=0 表示不裁剪)')

    # --- DDP 相关参数 ---
    parser.add_argument('--dist_url', default='env://', help='DDP 使用的 URL')
    
    args = parser.parse_args()
    return args

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
        
        with torch.amp.autocast(device_type='cuda'):
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
def evaluate(model, data_loader, device, args):
    """评估函数，用于验证集和测试集"""
    model.eval()
    correct_meter = torch.tensor(0.0, device=device)

    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.amp.autocast(device_type='cuda'):
            outputs = model(inputs)
        correct_meter += (outputs.argmax(1) == labels).sum()
    
    # DDP - 同步结果
    total_correct = reduce_value(correct_meter, average=False)
    
    total_samples = len(data_loader.sampler.dataset) if args.distributed else len(data_loader.dataset)
    accuracy = 100. * total_correct.item() / total_samples
    return accuracy

if __name__ == '__main__':
    arg = parameter_setting()
    init_distributed_mode(arg)
    
    if arg.rank == 0:
        os.makedirs('./results/', exist_ok=True)
        writer = SummaryWriter('runs/ConvNeXt_Up')

    torch.manual_seed(arg.seed)
    np.random.seed(arg.seed)
    device = torch.device(f"cuda:{arg.gpu}")

    train_transform, val_transform = get_data_transforms(arg.img_size)
    full_train_dataset = load_data(os.path.join(arg.data_path, 'train'), train_transform)
    test_set = load_data(os.path.join(arg.data_path, 'test'), val_transform)

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

    # 模型、优化器和调度器初始化
    model = convnext_1(num_classes=arg.classes).to(device)
    
    if arg.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[arg.gpu])
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=arg.lr, weight_decay=arg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=arg.warmup_epochs, T_mult=2, eta_min=arg.min_lr)
    scaler = torch.amp.GradScaler()

    dataset_name = os.path.basename(os.path.normpath(arg.data_path))
    best_ckpt_path = os.path.join('./results/', f'{dataset_name}_ConvNeXt_best.pth')
    
    best_val_acc, best_epoch, epochs_no_improve = 0, 0, 0

    # 训练循环
    for epoch in range(1, arg.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, scaler, arg)
        val_acc = evaluate(model, val_loader, device, arg)
        scheduler.step()

        if arg.rank == 0:
            print(f"[Epoch {epoch}] Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.2e}")
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/validation', val_acc, epoch)
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

            # --- 检查早停条件 ---
            # 1. 基于验证集性能
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                epochs_no_improve = 0
                model_to_save = model.module if arg.distributed else model
                torch.save(model_to_save.state_dict(), best_ckpt_path)
                print(f"*** 新最佳模型保存在 Epoch {best_epoch} (Val Acc: {best_val_acc:.2f}%) ***")
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= arg.patience:
                print(f"连续 {arg.patience} 轮验证集性能未提升，触发早停。")
                break
            
            # 2.基于训练/验证集差距
            if epoch > arg.warmup_epochs and (train_acc - val_acc > arg.overfit_gap_threshold):
                print(f"训练/验证准确率差距 ({train_acc - val_acc:.2f}%) 超过阈值 {arg.overfit_gap_threshold}，触发早停。")
                break

    if arg.distributed:
        dist.barrier()

    # 最终测试
    if arg.rank == 0:
        print(f"\n--- 训练结束, 使用 Epoch {best_epoch} 的最佳模型测试 ---")
        if os.path.exists(best_ckpt_path):
            test_model = convnext_1(num_classes=arg.classes).to(device)
            test_model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
            test_loader_single = DataLoader(test_set, batch_size=arg.batch_size, shuffle=False, num_workers=arg.workers)
            test_acc = model_test(test_model, test_loader_single)
            print(f"最终测试 OA: {test_acc:.2f}%")
        else:
            print("未找到最佳模型文件，跳过测试。")

    if arg.rank == 0:
        writer.close()

    cleanup()