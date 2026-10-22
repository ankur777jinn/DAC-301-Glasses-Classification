"""
DAC-301 Deep Learning Assignment: Glasses Classification
Extended 50-Epoch High-Performance Training Pipeline
Optimized for high-end GPUs (e.g., NVIDIA Blackwell / H100 / RTX 4090).
"""

import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

def get_args():
    parser = argparse.ArgumentParser(description="DAC-301 Glasses Classifier Training (50 Epochs)")
    parser.add_argument("--train_csv", type=str, default=r"train.csv", help="Path to train.csv")
    parser.add_argument("--train_dir", type=str, default=r".", help="Directory containing train images")
    parser.add_argument("--test_csv", type=str, default=r"test_dataset.csv", help="Path to test_dataset.csv")
    parser.add_argument("--test_dir", type=str, default=r"testimages", help="Directory containing test images")
    parser.add_argument("--sample_sub", type=str, default=r"sample_submission_dataset.csv", help="Sample submission CSV")
    parser.add_argument("--output_csv", type=str, default="extended_50ep_submission.csv", help="Output prediction CSV")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size (e.g. 128 for Blackwell/high-VRAM, 64 for 4GB)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Peak learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for AdamW")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader CPU workers")
    return parser.parse_args()

class GlassesDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, is_test=False):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = os.path.join(self.root_dir, str(self.df.iloc[idx]['file']))
        try:
            img = Image.open(path).convert('RGB')
        except Exception:
            img = Image.new('RGB', (224, 224), (0, 0, 0))
            
        if self.transform:
            img = self.transform(img)
            
        if self.is_test:
            return img, str(self.df.iloc[idx]['id'])
        return img, torch.tensor(int(self.df.iloc[idx]['glasses']), dtype=torch.long)

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

tta_flip_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_c)
        self.pool  = nn.MaxPool2d(2)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.pool(self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))))

class CustomGlassesNet(nn.Module):
    def __init__(self, num_hidden=2, sizes=[512, 256], activation='relu'):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3,   32),
            ConvBlock(32,  64),
            ConvBlock(64,  128),
            ConvBlock(128, 256),
            ConvBlock(256, 512),
        )
        in_f = 512 * 7 * 7
        act_map = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}
        act_fn = act_map.get(activation.lower(), nn.ReLU)
        
        layers = []
        for i in range(num_hidden):
            layers += [nn.Linear(in_f, sizes[i]), nn.BatchNorm1d(sizes[i]), act_fn(), nn.Dropout(0.4)]
            in_f = sizes[i]
        layers.append(nn.Linear(in_f, 2))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        return self.classifier(torch.flatten(self.features(x), 1))

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"==================================================")
    print(f"DAC-301 Glasses Classifier — 50 Epochs Training")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Batch Size: {args.batch_size} | LR: {args.lr} | Workers: {args.num_workers}")
    print(f"==================================================")

    # Dataset & Loader
    train_ds = GlassesDataset(args.train_csv, args.train_dir, transform=train_transform)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=False
    )
    print(f"Dataset Size: {len(train_ds):,} images | Batches per Epoch: {len(train_loader)}")

    # Model & Optimization
    model = CustomGlassesNet(num_hidden=2, sizes=[512, 256], activation='relu').to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 50 Epochs with Cosine Annealing with Warm Restarts (T_0=15, T_mult=1)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=15, T_mult=1, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

    best_loss = float('inf')
    best_weights_path = "best_glasses_model_50ep.pth"

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for i, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                out = model(imgs)
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * imgs.size(0)

        scheduler.step(epoch)
        avg_loss = running_loss / len(train_ds)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1:02d}/{args.epochs:02d}] | Loss: {avg_loss:.5f} | LR: {current_lr:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), best_weights_path)

    print(f"\nTraining Complete! Best Loss Achieved: {best_loss:.5f}")
    print(f"Loading checkpoint '{best_weights_path}' for Test-Time Augmentation (TTA)...")
    model.load_state_dict(torch.load(best_weights_path))
    model.eval()

    if os.path.exists(args.test_csv) and os.path.exists(args.test_dir):
        print("Running Test-Time Augmentation (TTA) on test images...")
        test_normal = GlassesDataset(args.test_csv, args.test_dir, transform=val_transform, is_test=True)
        test_flip   = GlassesDataset(args.test_csv, args.test_dir, transform=tta_flip_transform, is_test=True)
        
        ld_norm = DataLoader(test_normal, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        ld_flip = DataLoader(test_flip,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        all_ids, all_preds = [], []
        with torch.no_grad():
            for (x_n, ids), (x_f, _) in zip(ld_norm, ld_flip):
                x_n, x_f = x_n.to(device, non_blocking=True), x_f.to(device, non_blocking=True)
                p_n = torch.softmax(model(x_n), dim=1)
                p_f = torch.softmax(model(x_f), dim=1)
                avg_prob = (p_n + p_f) / 2.0
                _, pred = torch.max(avg_prob, 1)
                all_ids.extend(ids)
                all_preds.extend(pred.cpu().numpy())

        sub_df = pd.DataFrame({'id': all_ids, 'glasses': all_preds})
        sub_df['id'] = sub_df['id'].astype(str)
        if os.path.exists(args.sample_sub):
            sample = pd.read_csv(args.sample_sub)
            sample['id'] = sample['id'].astype(str)
            sub_df = sample[['id']].merge(sub_df, on='id', how='left')

        sub_df.to_csv(args.output_csv, index=False)
        print(f"Successfully generated and saved predictions to: {args.output_csv}")
    else:
        print(f"Test files not found at specified paths. Model weights saved to: {best_weights_path}")

if __name__ == '__main__':
    main()
