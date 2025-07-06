import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, LinearLR, SequentialLR

def create_scheduler(optimizer, args):
    """
    根据参数创建学习率调度器。

    支持的调度器类型:
    - 'warmup_cosine': 预热 + 余弦退火
    - 'cosine': 余弦退火
    - 'step': 步进学习率下降
    """
    
    scheduler_name = args.scheduler.lower()
    
    if scheduler_name == 'warmup_cosine':
        # "预热 + 余弦退火" 学习率调度器
        warmup_scheduler = LinearLR(
            optimizer, 
            start_factor=args.min_lr / args.lr if args.lr > 0 else 0.01, 
            total_iters=args.warmup_epochs
        )
        main_scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=args.epochs - args.warmup_epochs, 
            eta_min=args.min_lr
        )
        scheduler = SequentialLR(
            optimizer, 
            schedulers=[warmup_scheduler, main_scheduler], 
            milestones=[args.warmup_epochs]
        )
        return scheduler

    elif scheduler_name == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=args.epochs, 
            eta_min=args.min_lr
        )
        return scheduler

    elif scheduler_name == 'step':
        # 每 step_size 个 epoch，学习率乘以 gamma
        scheduler = StepLR(
            optimizer, 
            step_size=args.lr_step_size, 
            gamma=args.lr_gamma
        )
        return scheduler
        
    else:
        raise ValueError(f"不支持的学习率调度器类型: '{args.scheduler}'") 