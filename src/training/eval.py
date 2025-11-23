# src/training/eval.py
"""
Evaluate a trained model on a labeled CSV dataset.
Saves:
 - outputs/eval_metrics.json
 - outputs/confusion_matrix.png
 - outputs/roc_auc.png

Usage (project root):
export PYTHONPATH="$PWD"
python src/training/eval.py \
  --model outputs/best_model.pth \
  --csv data/processed/aptos/train_labels.csv \
  --img_dir data/processed/aptos/images \
  --img_size 224 \
  --batch_size 16
"""

import os
import json
import argparse
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize

from src.training.dataset import FundusDataset
from src.models.model import get_resnet50, CustomCNN

def predict_model(model, loader, device):
    model.eval()
    preds = []
    trues = []
    probs = []
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            out = model(x)
            prob = torch.nn.functional.softmax(out, dim=1).cpu().numpy()
            p = out.argmax(dim=1).cpu().numpy()
            preds.extend(p.tolist())
            trues.extend(y.numpy().tolist())
            probs.extend(prob.tolist())
    return np.array(preds), np.array(trues), np.array(probs)

def plot_confusion(cm, labels, out_path):
    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.yticks(range(len(labels)), labels)
    # annotate
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.colorbar()
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()

def plot_roc(y_true, y_prob, n_classes, out_path):
    # y_true: array shape (N,), y_prob: shape (N, n_classes)
    try:
        y_true_b = label_binarize(y_true, classes=list(range(n_classes)))
        # compute macro-average AUC
        aucs = []
        for i in range(n_classes):
            try:
                auc = roc_auc_score(y_true_b[:, i], y_prob[:, i])
            except Exception:
                auc = float('nan')
            aucs.append(auc)
        macro_auc = np.nanmean(aucs)
    except Exception:
        aucs = [float('nan')]*n_classes
        macro_auc = float('nan')

    plt.figure(figsize=(6,4))
    plt.bar(range(n_classes), [a if not np.isnan(a) else 0 for a in aucs])
    plt.xticks(range(n_classes), [str(i) for i in range(n_classes)])
    plt.xlabel("Class")
    plt.ylabel("ROC AUC (one-vs-rest)")
    plt.title(f"Per-class ROC AUC (macro {macro_auc:.3f})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return macro_auc, aucs

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # load CSV
    import pandas as pd
    df = pd.read_csv(args.csv)
    ds = FundusDataset(df, args.img_dir, mode='val', size=args.img_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # load model architecture & weights
    if args.backbone == 'resnet50':
        model = get_resnet50(n_classes=args.num_classes, pretrained=False)
    else:
        model = CustomCNN(n_classes=args.num_classes)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)

    preds, trues, probs = predict_model(model, loader, device)

    # metrics
    acc = accuracy_score(trues, preds)
    precision, recall, f1, support = precision_recall_fscore_support(trues, preds, labels=list(range(args.num_classes)), zero_division=0)
    cm = confusion_matrix(trues, preds, labels=list(range(args.num_classes)))

    # roc auc (one-vs-rest)
    try:
        macro_auc, per_class_auc = plot_roc(trues, probs, args.num_classes, os.path.join(args.output_dir, 'roc_auc.png'))
    except Exception:
        macro_auc = float('nan')
        per_class_auc = [float('nan')]*args.num_classes

    # save confusion matrix
    plot_confusion(cm, [str(i) for i in range(args.num_classes)], os.path.join(args.output_dir, 'confusion_matrix.png'))

    metrics = OrderedDict()
    metrics['accuracy'] = float(acc)
    metrics['macro_auc'] = float(macro_auc) if not np.isnan(macro_auc) else None
    metrics['per_class'] = {}
    for i in range(args.num_classes):
        metrics['per_class'][str(i)] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(support[i]),
            'roc_auc': (float(per_class_auc[i]) if (per_class_auc and not np.isnan(per_class_auc[i])) else None)
        }

    # Save json
    out_json = os.path.join(args.output_dir, 'eval_metrics.json')
    with open(out_json, 'w') as f:
        json.dump(metrics, f, indent=2)

    # print short summary
    print("Evaluation completed.")
    print("Accuracy:", metrics['accuracy'])
    print("Macro AUC:", metrics['macro_auc'])
    print("Per-class metrics:")
    for k,v in metrics['per_class'].items():
        print(f" Class {k}: precision={v['precision']:.3f} recall={v['recall']:.3f} f1={v['f1']:.3f} support={v['support']} roc_auc={v['roc_auc']}")

    print("Saved metrics ->", out_json)
    print("Saved confusion matrix ->", os.path.join(args.output_dir, 'confusion_matrix.png'))
    print("Saved ROC AUC plot ->", os.path.join(args.output_dir, 'roc_auc.png'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='path to model weights (.pth)')
    parser.add_argument('--csv', required=True, help='csv file with image, label')
    parser.add_argument('--img_dir', required=True, help='directory with images')
    parser.add_argument('--output_dir', default='outputs', help='where to save eval artifacts')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--backbone', choices=['resnet50','custom'], default='resnet50')
    args = parser.parse_args()
    main(args)
