import torch
import os
import numpy as np
import sys
sys.path.append('..')
from torchvision import models
from PIL import Image
from torch import nn
import torch.nn.functional as F
from torchvision import transforms as T
import random
import cv2
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
import matplotlib.pyplot as plt
from captum.attr import DeepLift, GuidedGradCam
from typing import Dict, Iterable, Callable
from tqdm import tqdm

class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self.features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, input, output):
            self.features[layer_id] = output
        return fn

    def forward(self, x):
        out = self.model(x)
        return out, self.features


def loss_fn(x, y):
    # criterion = nn.CosineSimilarity(dim=1)
    # -(criterion(p1, z2).mean()
    y = y.detach()
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    # print(criterion(x, y).shape)
    # return 2 - 2 * (x * y).sum(dim=-1)
    # return 1/2 - (criterion(x, y).mean())*1/2
    return 1/2 - (x * y).sum(dim=-1).mean()/2


class CapsuleLoss(nn.Module):
    """Combine margin loss & reconstruction loss of capsule network."""

    def __init__(self, upper_bound=0.9, lower_bound=0.1, lmda=0.5):
        super(CapsuleLoss, self).__init__()
        self.upper = upper_bound
        self.lower = lower_bound
        self.lmda = lmda

    def forward(self, logits, labels):
        # Shape of left / right / labels: (batch_size, num_classes)
        # left = (self.upper - logits).relu() ** 2  # True negative
        # right = (logits - self.lower).relu() ** 2  # False positive
        # labels = torch.zeros(logits.shape).cuda().scatter_(1, labels.unsqueeze(1), 1)
        # margin_loss = torch.sum(labels * left) + self.lmda * torch.sum((1 - labels) * right)

        left = (self.upper - logits).relu() ** 2  # True negative
        right = (logits - self.lower).relu() ** 2  # False positive
        # print(logits.shape)
        # print(labels.shape)
        labels = torch.zeros(logits.shape).cuda().scatter_(1, labels.unsqueeze(1), 1)
        margin_loss = (labels * left).sum(-1).mean() + self.lmda * ((1 - labels) * right).sum(-1).mean()

        # Reconstruction loss

        # Combine two losses
        return margin_loss


def model_train(model, data_loader, opt, sch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    cr1 = nn.CrossEntropyLoss()
    correct = 0
    total_loss = 0
    total = 0

    # 使用tqdm的动态进度条，设置position=0确保在同一行更新
    progress_bar = tqdm(data_loader, desc="Training", ncols=100, position=0, leave=True)
    
    for i, (x, y) in enumerate(progress_bar):
        x, y = x.to(device), y.to(device)
        batch_size = y.size(0)
        total += batch_size
        
        output = model(x)
        loss = cr1(output, y)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        pred = output.max(1, keepdim=True)[1]
        batch_correct = pred.eq(y.view_as(pred)).sum().item()
        correct += batch_correct
        total_loss += loss.item()
        
        # 更新进度条信息
        current_acc = 100. * correct / total
        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{current_acc:.2f}%"
        )
    
    # 每个epoch结束后更新学习率
    sch.step()
    
    # 最终的训练精度和损失
    avg_loss = total_loss / len(data_loader)
    accuracy = 100. * correct / total
    print(f"Train Accuracy is: {accuracy:.2f}%, Average Loss: {avg_loss:.4f}")
    return

def model_val(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    test_loss = 0
    correct = 0
    total = 0
    model.eval()
    
    # 使用简洁的进度条
    progress_bar = tqdm(test_loader, desc="Validation", ncols=100, position=0, leave=True)
    
    with torch.no_grad():
        for data, target in progress_bar:
            data, target = data.to(device), target.to(device)
            batch_size = target.size(0)
            total += batch_size
            
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            batch_correct = pred.eq(target.view_as(pred)).sum().item()
            correct += batch_correct
            
            # 更新进度条信息
            current_acc = 100. * correct / total
            progress_bar.set_postfix(acc=f"{current_acc:.2f}%")
    
    final_acc = 100. * correct / total
    return final_acc

def model_test(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    test_loss = 0
    correct = 0
    total = 0
    model.eval()
    
    # 使用简洁的进度条
    progress_bar = tqdm(test_loader, desc="Testing", ncols=100, position=0, leave=True)
    
    with torch.no_grad():
        for data, target in progress_bar:
            data, target = data.to(device), target.to(device)
            batch_size = target.size(0)
            total += batch_size
            
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            batch_correct = pred.eq(target.view_as(pred)).sum().item()
            correct += batch_correct
            
            # 更新进度条信息
            current_acc = 100. * correct / total
            progress_bar.set_postfix(acc=f"{current_acc:.2f}%")
    
    final_acc = 100. * correct / total
    return final_acc

