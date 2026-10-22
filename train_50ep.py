"""
DAC-301 Deep Learning Assignment: Glasses Classification
Extended 50-Epoch High-Performance Training Pipeline with Scalable Backbone Options.
Backbone Options:
  - standard : 5-stage VGG-style CNN (~18M params)
  - large    : 5-stage Wide/Deep VGG CNN with 3 convs per deep stage & 1024 channels (~58M params)
  - resnet   : From-scratch Custom ResNet-34 style CNN with residual connections & GAP (~21M params)
All models adhere 100% to competition rules (trained from scratch, pure PyTorch, no transfer learning, no ViTs/DDPMs).
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
    parser = argparse.ArgumentParser(description="DAC-301 Scaled Architecture Training (50 Epochs)")
    parser.add_argument("--model_type", type=str, default="large", choices=["standard", "large", "resnet"],
                        help="Model architecture: standard (18M), large (58M), or resnet (from scratch ResNet-34 style)")
    parser.add_argument("--train_csv", type=str, default=r"train.csv", help="Path to train.csv")
    parser.add_argument("--train_dir", type=str, default=r".", help="Directory containing train images")
    parser.add_argument("--test_csv", type=str, default=r"test_dataset.csv", help="Path to test_dataset.csv")
    parser.add_argument("--test_dir", type=str, default=r"testimages", help="Directory containing test images")
    parser.add_argument("--sample_sub", type=str, default=r"sample_submission_dataset.csv", help="Sample submission CSV")
    parser.add_argument("--output_csv", type=str, default="scaled_50ep_submission.csv", help="Output prediction CSV")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size (e.g. 128/256 on Blackwell/H100)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Peak learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for AdamW")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader CPU workers")
    return parser.parse_args()

# --- Dataset Definition ---
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

# Transforms with multi-scale perturbations
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

# --- Model 1: Standard VGG-style Net (18M params) ---
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, num_convs=2):
        super().__init__()
        layers = []
        curr_in = in_c
        for _ in range(num_convs):
            layers.append(nn.Conv2d(curr_in, out_c, 3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.ReLU(inplace=True))
            curr_in = out_c
        layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class CustomGlassesNet(nn.Module):
    def __init__(self, num_hidden=2, sizes=[512, 256], activation='relu'):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
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

# --- Model 2: Large Wide/Deep VGG-19 Style Net (~58M params) ---
class LargeGlassesNet(nn.Module):
    """Deeper and wider feature extractor scaling up to 1024 channels and 1024-512 MLP head."""
    def __init__(self, num_hidden=2, sizes=[1024, 512], activation='relu'):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 64, num_convs=2),      # 224 -> 112
            ConvBlock(64, 128, num_convs=2),    # 112 -> 56
            ConvBlock(128, 256, num_convs=3),   # 56  -> 28
            ConvBlock(256, 512, num_convs=3),   # 28  -> 14
            ConvBlock(512, 1024, num_convs=3),  # 14  -> 7
        )
        in_f = 1024 * 7 * 7  # 50,176 features
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

# --- Model 3: From-Scratch ResNet-34 Style Backbone (~21M params) ---
class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        res = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += res
        return self.relu(out)

class FromScratchResNet(nn.Module):
    """From-scratch ResNet-34 style architecture with skip connections (no pretrained weights)."""
    def __init__(self, num_classes=2, num_hidden=1, sizes=[512]):
        super().__init__()
        self.in_c = 64
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),  # 224 -> 112
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)                  # 112 -> 56
        )
        self.layer1 = self._make_layer(64, num_blocks=3, stride=1)   # 56 -> 56
        self.layer2 = self._make_layer(128, num_blocks=4, stride=2)  # 56 -> 28
        self.layer3 = self._make_layer(256, num_blocks=6, stride=2)  # 28 -> 14
        self.layer4 = self._make_layer(512, num_blocks=3, stride=2)  # 14 -> 7
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        in_f = 512
        layers = []
        for i in range(num_hidden):
            layers += [nn.Linear(in_f, sizes[i]), nn.BatchNorm1d(sizes[i]), nn.ReLU(inplace=True), nn.Dropout(0.4)]
            in_f = sizes[i]
        layers.append(nn.Linear(in_f, num_classes))
        self.classifier = nn.Sequential(*layers)

    def _make_layer(self, out_c, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(ResidualBlock(self.in_c, out_c, stride=s))
            self.in_c = out_c
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

def build_model(model_type):
    if model_type == "large":
        return LargeGlassesNet(num_hidden=2, sizes=[1024, 512])
    elif model_type == "resnet":
        return FromScratchResNet(num_classes=2, num_hidden=1, sizes=[512])
    else:
        return CustomGlassesNet(num_hidden=2, sizes=[512, 256])

# --- Main Execution Loop ---
def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 60)
    print(f"DAC-301 Glasses Classifier Training (50 Epochs)")
    print(f"Selected Backbone Architecture: [{args.model_type.upper()}]")
    print(f"Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Batch Size: {args.batch_size} | LR: {args.lr} | DataLoader Workers: {args.num_workers}")
    print("=" * 60)

    # Initialize model
    model = build_model(args.model_type).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Instantiated Model '{args.model_type}' with {total_params:,} parameters (All from scratch).")

    # DataLoader
    train_ds = GlassesDataset(args.train_csv, args.train_dir, transform=train_transform)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=False
    )
    print(f"Total Training Images: {len(train_ds):,} | Steps per Epoch: {len(train_loader)}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Cosine Annealing with Warm Restarts over 50 epochs (cycles every 15 epochs)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=15, T_mult=1, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

    best_loss = float('inf')
    ckpt_name = f"best_model_{args.model_type}_50ep.pth"

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
            torch.save(model.state_dict(), ckpt_name)

    print(f"\nTraining Complete! Best Training Loss: {best_loss:.5f}")
    print(f"Loading optimal checkpoint '{ckpt_name}' for Test-Time Augmentation (TTA)...")
    model.load_state_dict(torch.load(ckpt_name))
    model.eval()

    if os.path.exists(args.test_csv) and os.path.exists(args.test_dir):
        print("Generating final test predictions with TTA (Normal + Horizontal Flip)...")
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
        print(f"Predictions saved to: {args.output_csv}")
    else:
        print(f"Test files not found at specified paths. Model weights saved to: {ckpt_name}")

if __name__ == '__main__':
    main()
