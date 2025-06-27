import sys
import torch
import numpy as np
import re
from tqdm import tqdm
import argparse
import torch.nn as nn
import collections
from functools import partial
import torchvision.transforms as transforms
from utils.DataLoad import load_data
from utils.TrainTest import model_val, model_test, model_train
from model.HiVit import HiViT, HiViT_base
import os
import gc


# True: 只加载模型并测试
# False: 正常训练模型
TEST_ONLY = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def parameter_setting():
    # argparse settings
    parser = argparse.ArgumentParser(description='Origin Input')
    default_data_path = os.path.realpath(os.path.join(SCRIPT_DIR, '..', '..', '..', '..', '..', 'datasets', 'SOC_50classes/'))
    parser.add_argument('--data_path', type=str, default=default_data_path,
                        help='where data is stored')
    parser.add_argument('--GPU_ids', type=str, default='0,1',
                        help='GPU ids, comma-separated (e.g., 0,1)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train')
    parser.add_argument('--classes', type=int, default=50,
                        help='number of classes')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training')
    parser.add_argument('--workers', type=int, default=4,
                        help='number of data loading workers')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate')
    parser.add_argument('--fold', type=int, default=1,
                        help='K-fold')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 1)')
    parser.add_argument('--img_size', type=int, default=224,
                        help='input image size')
    args = parser.parse_args()
    return args

def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        new_size = int(num_patches ** 0.5)
        if orig_size != new_size:
            print(f"Position interpolate from {orig_size}x{orig_size} to {new_size}x{new_size}")
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed

def train_parallel(model, data_loader, optimizer, scheduler=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total_loss = 0
    total = 0
    
    progress_bar = tqdm(data_loader, desc="Training", ncols=100, leave=True)
    
    for i, (inputs, labels) in enumerate(progress_bar):
        inputs, labels = inputs.to(device), labels.to(device)
        batch_size = labels.size(0)
        total += batch_size
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        pred = outputs.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total_loss += loss.item()
        
        current_acc = 100. * correct / total
        progress_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{current_acc:.2f}%")

    if scheduler is not None:
        scheduler.step()
    
    avg_loss = total_loss / len(data_loader)
    accuracy = 100. * correct / total
    print(f"Train Accuracy: {accuracy:.2f}%, Average Loss: {avg_loss:.4f}")
    return avg_loss

def test_and_report(model, test_loader, best_epoch, best_test_accuracy, history):
    acc = model_test(model, test_loader)
    print(f'Final Test Accuracy is {acc}')
    
    if not history['accuracy']:
        history['accuracy'].append(acc)

    print(f'The best epoch during training was {best_epoch}, with val accuracy: {best_test_accuracy:.2f}%.')
    print(f'Final test accuracy on the loaded model is: {acc:.2f}%.')
    
    print(f"Overall Accuracy (OA) is {np.mean(history['accuracy'])}, STD is {np.std(history['accuracy'])}")
    print("All accuracy records: ", history['accuracy'])

if __name__ == '__main__':
    os.makedirs('./Model/', exist_ok=True)
    os.makedirs('./results/', exist_ok=True)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    arg = parameter_setting()
    gpu_ids = [int(id) for id in arg.GPU_ids.split(',')]
    device = torch.device(f"cuda:{gpu_ids[0]}" if torch.cuda.is_available() else "cpu")
    
    torch.backends.cudnn.benchmark = True
    
    history = collections.defaultdict(list)

    data_transform = transforms.Compose([
        transforms.Resize(arg.img_size),
        transforms.ToTensor(),
    ])

    try:
        model = HiViT_base(arg.classes)
        print("成功创建HiViT_base模型")
    except Exception as e:
        print(f"创建HiViT_base模型失败: {e}")
        model = HiViT(img_size=arg.img_size, num_classes=arg.classes)
        print("使用默认配置创建HiViT模型")

    if torch.cuda.is_available() and len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=arg.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=arg.epochs)
    best_test_accuracy = 0
    best_epoch = 0
    dataset_name = os.path.basename(os.path.normpath(arg.data_path))
    save_path = os.path.join('./Model/', f'{dataset_name}_HiViT.pth')

    if not TEST_ONLY:
        print("--- Starting Training ---\n")
        train_all = load_data(os.path.join(arg.data_path, 'train'), data_transform)
        test_set = load_data(os.path.join(arg.data_path, 'test'), data_transform)
        
        for k_F in tqdm(range(arg.fold), desc="Folds"):
            train_loader = torch.utils.data.DataLoader(
                train_all, batch_size=arg.batch_size, shuffle=True, 
                num_workers=arg.workers)
            test_loader = torch.utils.data.DataLoader(
                test_set, batch_size=arg.batch_size, shuffle=False, 
                num_workers=arg.workers)
            
            best_test_accuracy_fold = 0
            best_epoch_fold = 0

            for epoch in range(1, arg.epochs + 1):
                print(f"\n--- Fold {k_F + 1}, Epoch {epoch} ---\n")
                
                model_train(model, train_loader, opt, scheduler)
                
                accuracy = model_val(model, test_loader)
                print(f"--- Val Accuracy: {accuracy:.2f}% ---")

                if accuracy >= best_test_accuracy_fold:
                    best_epoch_fold = epoch
                    best_test_accuracy_fold = accuracy
                    state_to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                    torch.save(state_to_save, save_path)
                    print(f"*** New best model saved at epoch {best_epoch_fold} with accuracy {best_test_accuracy_fold:.2f}% ***")
            
            best_test_accuracy = best_test_accuracy_fold
            best_epoch = best_epoch_fold
            
            print(f"\nLoading best model from fold {k_F+1} (epoch {best_epoch}) for testing...")
            if os.path.exists(save_path):
                state_dict = torch.load(save_path)
                if isinstance(model, nn.DataParallel):
                    model.module.load_state_dict(state_dict)
                else:
                    model.load_state_dict(state_dict)
            acc = model_test(model, test_loader)
            history['accuracy'].append(acc)

    if TEST_ONLY:
        print("\n--- Testing Phase ---")
        if os.path.exists(save_path):
            state_dict = torch.load(save_path)
            if isinstance(model, nn.DataParallel):
                model.module.load_state_dict(state_dict)
            else:
                model.load_state_dict(state_dict)
            print("Model loaded successfully.")
            
            test_set = load_data(os.path.join(arg.data_path, 'test'), data_transform)
            test_loader = torch.utils.data.DataLoader(
                test_set, batch_size=arg.batch_size, shuffle=False,
                num_workers=arg.workers)
            test_and_report(model, test_loader, 0, 0, history)
        else:
            print("Error: Model file not found. Cannot run test.")

    if history['accuracy']:
        np.save(f'./results/{dataset_name}_result.npy', np.array(history['accuracy']))
    
    print(f"Best model checkpoint is saved at: {save_path}")