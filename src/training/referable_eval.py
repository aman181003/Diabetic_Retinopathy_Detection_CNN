# src/training/referable_eval.py
"""
Binary evaluation: referable DR (class >=2) vs non-referable (0-1).
Usage:
export PYTHONPATH="$PWD"
python src/training/referable_eval.py --model outputs/best_model.pth --csv data/processed/aptos/train_labels.csv --img_dir data/processed/aptos/images
"""
import argparse, os, json
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
import torch
from torch.utils.data import DataLoader
from src.training.dataset import FundusDataset
from src.models.model import get_resnet50, CustomCNN

def predict(model, loader, device):
    model.eval()
    preds=[]; trues=[]; probs=[]
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            out = model(x)
            p = out.argmax(dim=1).cpu().numpy()
            prob = torch.nn.functional.softmax(out, dim=1).cpu().numpy()
            preds.extend(p.tolist()); trues.extend(y.numpy().tolist()); probs.extend(prob.tolist())
    return np.array(preds), np.array(trues), np.array(probs)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    import pandas as pd
    df = pd.read_csv(args.csv)
    ds = FundusDataset(df, args.img_dir, mode='val', size=args.img_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = get_resnet50(n_classes=args.num_classes, pretrained=False) if args.backbone=='resnet50' else CustomCNN(n_classes=args.num_classes)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    preds, trues, probs = predict(model, loader, device)

    # binarize: referable = label >= 2
    preds_bin = (preds >= 2).astype(int)
    trues_bin = (trues >= 2).astype(int)
    acc = accuracy_score(trues_bin, preds_bin)
    precision, recall, f1, support = precision_recall_fscore_support(trues_bin, preds_bin, average='binary', zero_division=0)
    cm = confusion_matrix(trues_bin, preds_bin)

    # roc auc: use probability of referable = sum probs for classes >=2
    probs_refer = probs.sum(axis=1)[:,] if probs is not None else None
    try:
        auc = roc_auc_score(trues_bin, probs_refer)
    except:
        auc = None

    out = {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": (float(auc) if auc is not None else None),
        "confusion_matrix": cm.tolist()
    }
    os.makedirs(os.path.dirname(args.output) or "outputs", exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(out, f, indent=2)
    print("Referable evaluation saved to", args.output)
    print(json.dumps(out, indent=2))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--csv', required=True)
    parser.add_argument('--img_dir', required=True)
    parser.add_argument('--output', default='outputs/referable_metrics.json')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--backbone', choices=['resnet50','custom'], default='resnet50')
    args = parser.parse_args()
    main(args)
