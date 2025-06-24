import sys
sys.path.append('..')
import torch
import numpy as np
import re
from tqdm import tqdm
import argparse
import torch.nn as nn

import collections
from functools import partial
import torchvision.transforms as transforms
from utils.DataLoad import load_data, data_transform
from utils.TrainTest import model_train, model_val, model_test
from model.ConvNeXt import convnext_1
import os

# True: 只加载模型并测试
# False: 正常训练模型
TEST_ONLY = True

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
    parser.add_argument('--batch_size', type=int, default=512,
                        help='input batch size for training')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate')
    parser.add_argument('--fold', type=int, default=1,
                        help='K-fold')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 1)')
    args = parser.parse_args()
    return args

def test_and_report(model, test_loader, best_epoch, best_test_accuracy, history):
    """封装测试和报告逻辑"""
    acc = model_test(model, test_loader)
    print('Final Test Accuracy is {}'.format(acc))
    
    if not history['accuracy']:
        history['accuracy'].append(acc)

    print('The best epoch during training was {}, with val accuracy: {:.2f}%.'.format(best_epoch, best_test_accuracy))
    print('Final test accuracy on the loaded model is: {:.2f}%.'.format(acc))
    
    print('Overall Accuracy (OA) is {}, STD is {}'.format(np.mean(history['accuracy']), np.std(history['accuracy'])))
    print("All accuracy records: ", history['accuracy'])

if __name__ == '__main__':
    arg = parameter_setting()
    gpu_ids = [int(id) for id in arg.GPU_ids.split(',')]
    device = torch.device(f"cuda:{gpu_ids[0]}" if torch.cuda.is_available() else "cpu")
    # torch.manual_seed(arg.seed)
    # torch.cuda.manual_seed(arg.seed)
    history = collections.defaultdict(list)  # 记录每一折的各种指标

    model = convnext_1(arg.classes)

    # 为多卡训练包装模型
    if torch.cuda.is_available() and len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
    model.to(device)

    # opt = torch.optim.SGD(model.parameters(), momentum=0.9, lr=arg.lr, weight_decay=4e-3)
    # opt = torch.optim.Adam(model.parameters(), lr=arg.lr, weight_decay=0.004)
    opt = torch.optim.AdamW(model.parameters(), lr=arg.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=arg.epochs)
    best_test_accuracy = 0
    best_epoch = 0
    dataset_name = os.path.basename(os.path.normpath(arg.data_path))
    save_path = os.path.join('./Model/', f'{dataset_name}_ConvNeXt.pth')

    if not TEST_ONLY:
        # --- 训练模式 ---
        print("--- Starting Training ---\n")
        train_all = load_data(os.path.join(arg.data_path, 'train'), data_transform)
        test_set = load_data(os.path.join(arg.data_path, 'test'), data_transform)
        for k_F in tqdm(range(arg.fold), position=0, leave=True):
            train_loader = torch.utils.data.DataLoader(train_all, batch_size=arg.batch_size, shuffle=True)
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=arg.batch_size, shuffle=False)
            
            opt = torch.optim.AdamW(model.parameters(), lr=arg.lr, weight_decay=1e-3)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=arg.epochs)
            best_test_accuracy_fold = 0
            best_epoch_fold = 0

            for epoch in range(1, arg.epochs + 1):
                print("\n##### Fold {} EPOCH {} #####\n".format(k_F + 1, epoch))
                model_train(model=model, data_loader=train_loader, opt=opt, sch=scheduler)
                
                accuracy = model_val(model, test_loader)
                print("--- Epoch {} Val Accuracy is: {:.2f} % ---".format(epoch, accuracy))

                if best_test_accuracy_fold <= accuracy:
                    best_epoch_fold = epoch
                    best_test_accuracy_fold = accuracy
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    if isinstance(model, torch.nn.DataParallel):
                        torch.save(model.module.state_dict(), save_path)
                    else:
                        torch.save(model.state_dict(), save_path)
                    print(f"*** New best model for fold {k_F+1} saved at epoch {best_epoch_fold} with accuracy {best_test_accuracy_fold:.2f}% ***")
            
            best_test_accuracy = best_test_accuracy_fold
            best_epoch = best_epoch_fold
            print(f"\nLoading best model from fold {k_F+1} (epoch {best_epoch}) for testing...")
            if os.path.exists(save_path):
                if isinstance(model, torch.nn.DataParallel):
                    model.module.load_state_dict(torch.load(save_path))
                else:
                    model.load_state_dict(torch.load(save_path))
            acc = model_test(model, test_loader)
            history['accuracy'].append(acc)

    # --- 测试阶段 ---
    print("\n--- Testing Phase ---")
    print(f"Loading model from: {save_path}")
    if os.path.exists(save_path):
        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(torch.load(save_path))
        else:
            model.load_state_dict(torch.load(save_path))
        print("Model loaded successfully.")
    else:
        print("Error: Model file not found. Cannot run test.")
        sys.exit(1)

    test_set = load_data(os.path.join(arg.data_path, 'test'), data_transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=arg.batch_size, shuffle=False)
    
    test_and_report(model, test_loader, best_epoch, best_test_accuracy, history)

    print(f"Best model checkpoint is saved at: {save_path}")
