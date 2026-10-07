import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from dataset import make_dataset, one_hot
from model import DNAConvNet


def test_one_hot_shape():
    seqs, _ = make_dataset(n=10, seq_len=50)
    x = one_hot(seqs)
    assert x.shape == (10, 4, 50)
    assert set(x.reshape(-1).tolist()).issubset({0.0, 1.0})


def test_model_shape():
    model = DNAConvNet()
    x = torch.randn(4, 4, 200)
    y = model(x)
    assert y.shape == (4,)
