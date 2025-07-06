import torch
import torch.nn as nn
import os
import warnings
from typing import Dict, Any
import utils

def adapt_and_load_state_dict(model: torch.nn.Module, state_dict: Dict[str, Any], strict: bool = False):
    """
    一个通用的函数，用于在加载 state_dict 前进行适配。
    它会移除 'module.' 前缀。
    """
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    return model.load_state_dict(new_state_dict, strict=strict)

def load_finetuned_checkpoint(model: nn.Module, checkpoint_path: str):
    """
    加载已经微调过的检查点，包含模型、优化器、调度器等状态。
    - 它会特殊处理原始脚本中为了适配特定检查点而加入的 `BatchNorm1d` 层。
    """
    if utils.is_main_process():
        print(f"--- 正在加载微调检查点: {os.path.basename(checkpoint_path)} ---")

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # 修正: 检查点中模型权重的键是 'model_state_dict' 而不是 'model'
    if 'model_state_dict' in checkpoint:
        model_state_dict = checkpoint['model_state_dict']
    else:
        # 兼容旧格式或其他只保存模型权重的检查点
        model_state_dict = checkpoint.get('model', checkpoint)


    # 关键适配: 因为检查点是在包含BN层的head上训练的，所以先修改模型结构
    if 'head.0.running_mean' in model_state_dict and isinstance(model.head, nn.Linear):
        if utils.is_main_process():
            print("适配模型...")
        model.head = torch.nn.Sequential(
            torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6),
            model.head
        )

    load_result = adapt_and_load_state_dict(model, model_state_dict, strict=True)
    
    if utils.is_main_process():
        print("--- 微调检查点加载成功 ---")
    return checkpoint

def load_mae_pretrained_weights(model: nn.Module, checkpoint_path: str):
    """
    Loads MAE pre-trained weights into the model.

    This function handles the architectural differences between the MAE
    pre-training model and the classification fine-tuning model. Specifically:
    - It ignores the MAE model's `mask_token`, which is not needed for fine-tuning.
    - It correctly maps the `head` weights from the pre-trained model to the
      classification model's `head`, even if they have different names.
    - It uses `strict=False` to safely ignore any other layers that don't match,
      such as the MAE decoder's final norm layer or other components.
    """
    if not os.path.exists(checkpoint_path):
        print(f"--- MAE 预训练权重文件不存在: {checkpoint_path}，将跳过加载。 ---")
        return

    print(f"--- 正在从 {os.path.basename(checkpoint_path)} 加载 MAE 预训练权重 ---")
    
    # 加载 MAE 权重
    mae_checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    mae_state_dict = mae_checkpoint.get('model', mae_checkpoint)

    # 移除 'module.' 前缀 (DDP)
    mae_state_dict = {k.replace('module.', ''): v for k, v in mae_state_dict.items()}

    # 忽略 MAE 的 mask_token，因为它在分类模型中不存在
    mae_state_dict.pop('mask_token', None)
    

    load_result = model.load_state_dict(mae_state_dict, strict=False)

    if utils.is_main_process():
        print("--- MAE 权重加载完成。请检查以下不匹配的键: ---")
        if load_result.missing_keys:
            print("模型中存在但权重文件中缺失的键 (将被随机初始化):")
            # 通常我们期望看到 fc_norm 和 head
            print(f"  {', '.join(load_result.missing_keys)}")
        if load_result.unexpected_keys:
            print("权重文件中存在但模型中缺失的键 (将被忽略):")
            # 通常我们期望看到 norm 和 mask_token
            print(f"  {', '.join(load_result.unexpected_keys)}")
        print("--------------------------------------------------") 