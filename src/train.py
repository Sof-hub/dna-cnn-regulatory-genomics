import argparse
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score

from dataset import make_dataset, one_hot
from model import DNAConvNet
from utils import seed_everything


def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(device))
            ys.append(yb.numpy())
            ps.append(torch.sigmoid(logits).cpu().numpy())
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    pred = (p >= 0.5).astype(int)
    return {
        'roc_auc': float(roc_auc_score(y, p)),
        'average_precision': float(average_precision_score(y, p)),
        'accuracy': float(accuracy_score(y, pred)),
        'precision': float(precision_score(y, pred, zero_division=0)),
        'recall': float(recall_score(y, pred, zero_division=0)),
        'f1': float(f1_score(y, pred, zero_division=0)),
    }


def main(args):
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    seqs, y = make_dataset(args.n_samples, args.seq_len, seed=args.seed)
    idx = np.arange(len(y))
    train_idx, temp_idx = train_test_split(idx, test_size=0.30, stratify=y, random_state=args.seed)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, stratify=y[temp_idx], random_state=args.seed)

    X = one_hot(seqs)
    X = torch.from_numpy(X)
    Y = torch.from_numpy(y)
    loaders = {
        'train': DataLoader(TensorDataset(X[train_idx], Y[train_idx]), batch_size=args.batch_size, shuffle=True),
        'val': DataLoader(TensorDataset(X[val_idx], Y[val_idx]), batch_size=args.batch_size),
        'test': DataLoader(TensorDataset(X[test_idx], Y[test_idx]), batch_size=args.batch_size),
    }

    model = DNAConvNet().to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_state, best_auc = None, -np.inf

    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in loaders['train']:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
        val_metrics = evaluate(model, loaders['val'], device)
        print(f"epoch={epoch:02d} val_auc={val_metrics['roc_auc']:.4f} val_auprc={val_metrics['average_precision']:.4f}")
        if val_metrics['roc_auc'] > best_auc:
            best_auc = val_metrics['roc_auc']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    metrics = evaluate(model, loaders['test'], device)
    print(json.dumps(metrics, indent=2))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / 'dna_cnn.pt')
    with open(out / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    with open(out / 'config.json', 'w') as f:
        json.dump(vars(args) | {'device': str(device)}, f, indent=2)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--n-samples', type=int, default=12000)
    p.add_argument('--seq-len', type=int, default=200)
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--out-dir', default='results')
    main(p.parse_args())
