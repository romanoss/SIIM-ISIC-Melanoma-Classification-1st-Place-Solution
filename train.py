import os
import time
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
#from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
from util import GradualWarmupSchedulerV2
# import apex
# from apex import amp
from dataset import get_df, get_transforms, MelanomaDataset
from models import Effnet_Melanoma, Resnest_Melanoma, Seresnext_Melanoma

from lr_find import lr_finder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel-type', type=str, required=True)
    parser.add_argument('--data-dir', type=str, default='/raid/')
    parser.add_argument('--data-folder', type=int, required=True)
    parser.add_argument('--image-size', type=int, required=True)
    parser.add_argument('--enet-type', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=32)
    parser.add_argument('--init-lr', type=float, default=3e-5)
    parser.add_argument('--out-dim', type=int, default=9)
    parser.add_argument('--n-epochs', type=int, default=15)
    parser.add_argument('--use-amp', action='store_true')
    parser.add_argument('--use-meta', action='store_true')
    parser.add_argument('--DEBUG', action='store_true')
    parser.add_argument('--model-dir', type=str, default='./weights')
    parser.add_argument('--log-dir', type=str, default='./logs')
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='0')
    parser.add_argument('--fold', type=str, default='0,1,2,3,4')
    parser.add_argument('--n-meta-dim', type=str, default='512,128')

    args, _ = parser.parse_known_args()
    return args


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def upsample_train(df_train, upsample_times=16):
    df_mela = df_train[df_train["target"] == 6]
    # return df_train with concatenated df_mela for 16 times - mix it in
    if df_mela.empty:
        raise ValueError("No rows with target == 6 found in df_train")

    # Concatenate multiple copies of melanoma samples
    df_upsampled = pd.concat([df_train] + [df_mela.copy() for _ in range(upsample_times)], ignore_index=True)

    # Shuffle to mix up the duplicates with other samples
    df_upsampled = df_upsampled.sample(frac=1.0, random_state=42).reset_index(drop=True)

    return df_upsampled


def train_epoch(model, loader, optimizer, scaler):
    """
    Train one epoch using native PyTorch AMP (no Apex).
    Keeps the same behavior as the original Apex version.
    """
    model.train()
    train_loss = []

    # scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    bar = tqdm(loader)

    for (data, target) in bar:
        optimizer.zero_grad()

        # handle meta features if present
        if args.use_meta:
            data, meta = data
            data, meta, target = data.to(device), meta.to(device), target.to(device)
        else:
            data, target = data.to(device), target.to(device)
            meta = None

        # forward pass under autocast
        with torch.amp.autocast('cuda'):
            logits = model(data, meta) if args.use_meta else model(data)
            loss = criterion(logits, target)

        # backward pass (scaling if AMP enabled)
        if args.use_amp:
            scaler.scale(loss).backward()
            if args.image_size in [896, 576]:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.image_size in [896, 576]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                
        optimizer.step()

        # bookkeeping
        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        bar.set_description(f"loss: {loss_np:.5f}, smth: {smooth_loss:.5f}")

    return float(np.mean(train_loss))


def get_trans(img, I):

    if I >= 4:
        img = img.transpose(2, 3)
    if I % 4 == 0:
        return img
    elif I % 4 == 1:
        return img.flip(2)
    elif I % 4 == 2:
        return img.flip(3)
    elif I % 4 == 3:
        return img.flip(2).flip(3)


def val_epoch(model, loader, mel_idx, is_ext=None, n_test=1, get_output=False):

    model.eval()
    val_loss = []
    LOGITS = []
    PROBS = []
    TARGETS = []
    with torch.no_grad():
        for (data, target) in tqdm(loader):
            
            if args.use_meta:
                data, meta = data
                data, meta, target = data.to(device), meta.to(device), target.to(device)
                logits = torch.zeros((data.shape[0], args.out_dim)).to(device)
                probs = torch.zeros((data.shape[0], args.out_dim)).to(device)
                for I in range(n_test):
                    with torch.amp.autocast('cuda'):
                        l = model(get_trans(data, I), meta)
                    logits += l
                    probs += l.softmax(1)
            else:
                data, target = data.to(device), target.to(device)
                logits = torch.zeros((data.shape[0], args.out_dim)).to(device)
                probs = torch.zeros((data.shape[0], args.out_dim)).to(device)
                for I in range(n_test):
                    with torch.amp.autocast('cuda'):
                        l = model(get_trans(data, I))
                    logits += l
                    probs += l.softmax(1)
            logits /= n_test
            probs /= n_test

            LOGITS.append(logits.detach().cpu())
            PROBS.append(probs.detach().cpu())
            TARGETS.append(target.detach().cpu())

            loss = criterion(logits, target)
            val_loss.append(loss.detach().cpu().numpy())

    val_loss = np.mean(val_loss)
    LOGITS = torch.cat(LOGITS).numpy()
    PROBS = torch.cat(PROBS).numpy()
    TARGETS = torch.cat(TARGETS).numpy()

    if get_output:
        return LOGITS, PROBS
    else:
        acc = (PROBS.argmax(1) == TARGETS).mean() * 100.
        auc = roc_auc_score((TARGETS == mel_idx).astype(float), PROBS[:, mel_idx])
        auc_20 = roc_auc_score((TARGETS[is_ext == 0] == mel_idx).astype(float), PROBS[is_ext == 0, mel_idx])
        return val_loss, acc, auc, auc_20


def run(fold, df, meta_features, n_meta_features, transforms_train, transforms_val, mel_idx):

    if args.DEBUG:
        args.n_epochs = 5
        df_train = df[df['fold'] != fold].sample(args.batch_size * 5)
        df_valid = df[df['fold'] == fold].sample(args.batch_size * 5)
    else:
        df_train = df[df['fold'] != fold]
        df_valid = df[df['fold'] == fold]

    upsample_times=0
    if upsample_times > 0:
        print(f"upsampling df_train melanomas times {upsample_times} len {len(df_train)}")
        df_train = upsample_train(df_train, upsample_times=upsample_times) # orig psample_times=16
        print(f"upsampled df_train melanomas times {upsample_times} len {len(df_train)}")
    print("df_train target value counts", df_train["target"].value_counts())
    print()

    dataset_train = MelanomaDataset(df_train, 'train', meta_features, transform=transforms_train)
    dataset_valid = MelanomaDataset(df_valid, 'valid', meta_features, transform=transforms_val)
    train_loader = torch.utils.data.DataLoader(
        dataset_train, 
        batch_size=args.batch_size, 
        sampler=RandomSampler(dataset_train), 
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batch_size, num_workers=args.num_workers)

    model = ModelClass(
        args.enet_type,
        n_meta_features=n_meta_features,
        n_meta_dim=[int(nd) for nd in args.n_meta_dim.split(',')],
        out_dim=args.out_dim,
        pretrained=True
    )
    if DP:
        # model = apex.parallel.convert_syncbn_model(model)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)

    auc_max = 0.
    auc_20_max = 0.
    model_file  = os.path.join(args.model_dir, f'{args.kernel_type}_best_fold{fold}.pth')
    model_file2 = os.path.join(args.model_dir, f'{args.kernel_type}_best_20_fold{fold}.pth')
    model_file3 = os.path.join(args.model_dir, f'{args.kernel_type}_final_fold{fold}.pth')

    optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
    #if args.use_amp:
    #    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    #if DP:
    #    model = nn.DataParallel(model)

    # Native PyTorch AMP setup
    scaler = torch.amp.GradScaler('cuda')
    

    # Optional DataParallel wrapping (multi-GPU)
    if DP:
        model = nn.DataParallel(model)

#     scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs - 1)
    #scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.n_epochs - 1)
    #scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=1, after_scheduler=scheduler_cosine)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=int(args.n_epochs * 1.4),  
            T_mult=1
        )
    
    print(len(dataset_train), len(dataset_valid))

    print("lr_finder")
    lr_finder(model, train_loader, optimizer, criterion, device, start_lr=1e-8, end_lr=10, num_iters=100)

    for epoch in range(1, args.n_epochs + 1):
        print(time.ctime(), f'Fold {fold}, Epoch {epoch}')
#         scheduler_warmup.step(epoch - 1)

        train_loss = train_epoch(model, train_loader, optimizer, scaler)
        val_loss, acc, auc, auc_20 = val_epoch(model, valid_loader, mel_idx, is_ext=df_valid['is_ext'].values)

        content = time.ctime() + ' ' + f'Fold {fold}, Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {train_loss:.5f}, valid loss: {(val_loss):.5f}, acc: {(acc):.4f}, auc: {(auc):.6f}, auc_20: {(auc_20):.6f}.'
        print(content)
        with open(os.path.join(args.log_dir, f'log_{args.kernel_type}.txt'), 'a') as appender:
            appender.write(content + '\n')

        #scheduler_warmup.step()    
        #if epoch==2: scheduler_warmup.step() # bug workaround 
        scheduler.step()  

        print(f"Epoch {epoch}: LR={scheduler.get_last_lr()[0]:.6f}")
            
        if auc > auc_max:
            print('auc_max ({:.6f} --> {:.6f}). Saving model ...'.format(auc_max, auc))
            torch.save(model.state_dict(), model_file)
            auc_max = auc
        if auc_20 > auc_20_max:
            print('auc_20_max ({:.6f} --> {:.6f}). Saving model ...'.format(auc_20_max, auc_20))
            torch.save(model.state_dict(), model_file2)
            auc_20_max = auc_20

    torch.save(model.state_dict(), model_file3)


def main():

    print("args")
    print(args)
    print()
    
    df, df_test, meta_features, n_meta_features, mel_idx = get_df(
        args.kernel_type,
        args.out_dim,
        args.data_dir,
        args.data_folder,
        args.use_meta
    )

    transforms_train, transforms_val = get_transforms(args.image_size)

    folds = [int(i) for i in args.fold.split(',')]
    for fold in folds:
        run(fold, df, meta_features, n_meta_features, transforms_train, transforms_val, mel_idx)


if __name__ == '__main__':

    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES

    if args.enet_type == 'resnest101':
        ModelClass = Resnest_Melanoma
    elif args.enet_type == 'seresnext101':
        ModelClass = Seresnext_Melanoma
    elif 'efficientnet' in args.enet_type:
        ModelClass = Effnet_Melanoma
    else:
        raise NotImplementedError()

    #DP = len(os.environ['CUDA_VISIBLE_DEVICES']) > 1
    DP = len(args.CUDA_VISIBLE_DEVICES.split(',')) > 1

    set_seed()

    device = torch.device('cuda')
    criterion = nn.CrossEntropyLoss()

    main()
