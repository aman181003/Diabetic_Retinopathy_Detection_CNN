# src/training/train.py
"""
Training script with optional class-weights, WeightedRandomSampler oversampling, and Focal Loss.

Usage examples:
# baseline (no class weights, no sampler)
python src/training/train.py --csv data/processed/aptos/train_labels.csv --img_dir data/processed/aptos/images --output_dir outputs --epochs 10 --batch_size 16

# use class weights computed from training split
python src/training/train.py --csv data/processed/aptos/train_labels.csv --img_dir data/processed/aptos/images --output_dir outputs --epochs 10 --batch_size 16 --use_class_weights

# use sampler (oversample minority classes)
python src/training/train.py ... --use_sampler

# use focal loss + class weights
python src/training/train.py ... --use_focal --use_class_weights
"""
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from collections import Counter

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, WeightedRandomSampler

import mlflow

from src.training.dataset import FundusDataset
from src.models.model import get_resnet50, CustomCNN

# --- optional focal loss implementation ---
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce = nn.functional.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

# ---------- training & validation loops ----------
def train_loop(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        n += batch_size
    return total_loss / (n + 1e-12)

def val_loop(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0.0
    preds = []
    trues = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            preds.extend(out.argmax(dim=1).cpu().numpy().tolist())
            trues.extend(y.cpu().numpy().tolist())
    # compute per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(trues, preds, labels=list(range(num_classes)), zero_division=0)
    return total_loss / (len(trues) + 1e-12), np.array(preds), np.array(trues), precision, recall, f1, support

# ---------- main ----------
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    # read csv and split
    df = pd.read_csv(args.csv)
    train_df, val_df = train_test_split(df, test_size=args.val_split, stratify=df['label'], random_state=args.seed)
    print("Train / Val sizes:", len(train_df), len(val_df))
    os.makedirs(args.output_dir, exist_ok=True)

    # Datasets
    train_ds = FundusDataset(train_df, args.img_dir, mode='train', size=args.img_size)
    val_ds = FundusDataset(val_df, args.img_dir, mode='val', size=args.img_size)

    # optional sampler setup (compute sample weights)
    train_loader = None
    val_loader = None
    if args.use_sampler:
        print("Using WeightedRandomSampler for oversampling minority classes.")
        counts = Counter(train_df['label'].tolist())
        sample_weights = [1.0 / counts[int(l)] for l in train_df['label'].tolist()]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # model
    model = get_resnet50(n_classes=args.num_classes, pretrained=True) if args.backbone == 'resnet50' else CustomCNN(n_classes=args.num_classes)
    model = model.to(device)

    # compute class weights if requested
    class_weights = None
    if args.use_class_weights:
        from sklearn.utils.class_weight import compute_class_weight
        y = train_df['label'].values
        cw = compute_class_weight(class_weight='balanced', classes=np.arange(args.num_classes), y=y)
        class_weights = torch.tensor(cw, dtype=torch.float).to(device)
        print("Class weights:", cw)

    # criterion selection
    if args.use_focal:
        print("Using Focal Loss (gamma=2.0)")
        criterion = FocalLoss(gamma=2.0, weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3)

    # mlflow experiment
    mlflow.set_experiment(args.experiment)
    with mlflow.start_run():
        mlflow.log_params(vars(args))
        best_val_loss = float('inf')
        for epoch in range(args.epochs):
            train_loss = train_loop(model, train_loader, criterion, optimizer, device)
            val_loss, preds, trues, precision, recall, f1, support = val_loop(model, val_loader, criterion, device, args.num_classes)
            scheduler.step(val_loss)

            # log metrics per epoch
            mlflow.log_metric("train_loss", float(train_loss), step=epoch)
            mlflow.log_metric("val_loss", float(val_loss), step=epoch)
            # log per-class recall to mlflow for monitoring (as separate metrics)
            for i, r in enumerate(recall):
                mlflow.log_metric(f"recall_class_{i}", float(r), step=epoch)

            print(f"Epoch {epoch} | train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
            print("Per-class recall:", {i: float(r) for i,r in enumerate(recall)})
            # save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                out_path = os.path.join(args.output_dir, "best_model.pth")
                torch.save(model.state_dict(), out_path)
                print("Saved best model to", out_path)
                mlflow.log_artifact(out_path)

    print("Training finished. Best val loss:", best_val_loss)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--img_dir', required=True)
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--backbone', choices=['resnet50','custom'], default='resnet50')
    parser.add_argument('--use_class_weights', action='store_true', help='Compute class weights and use them in loss')
    parser.add_argument('--use_sampler', action='store_true', help='Use WeightedRandomSampler to oversample minority classes')
    parser.add_argument('--use_focal', action='store_true', help='Use focal loss instead of cross-entropy')
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--experiment', default='dr-detection')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(args)
