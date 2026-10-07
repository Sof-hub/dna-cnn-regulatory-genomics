import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from dataset import one_hot
from model import DNAConvNet


def mutate_sequence(seq, position, base):
    s = list(seq)
    s[position] = base
    return ''.join(s)


def in_silico_mutagenesis(model, sequence, device):
    bases = 'ACGT'
    original = one_hot([sequence])
    with torch.no_grad():
        ref = torch.sigmoid(model(torch.from_numpy(original).to(device))).item()
    scores = []
    for pos in range(len(sequence)):
        for base in bases:
            if base == sequence[pos]:
                continue
            mutant = mutate_sequence(sequence, pos, base)
            with torch.no_grad():
                score = torch.sigmoid(model(torch.from_numpy(one_hot([mutant])).to(device))).item()
            scores.append((pos, sequence[pos], base, score - ref))
    return ref, scores


def main(args):
    device = torch.device('cpu')
    model = DNAConvNet().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    ref, scores = in_silico_mutagenesis(model, args.sequence, device)
    best = sorted(scores, key=lambda x: abs(x[3]), reverse=True)[:20]
    print(f'reference_probability={ref:.4f}')
    for row in best:
        print(f'position={row[0]:03d} {row[1]}>{row[2]} delta={row[3]:+.4f}')

    pos_effect = np.zeros(len(args.sequence))
    for pos, _, _, delta in scores:
        pos_effect[pos] = max(pos_effect[pos], abs(delta))
    plt.figure(figsize=(12, 3))
    plt.plot(pos_effect)
    plt.xlabel('Sequence position')
    plt.ylabel('Max |Δ prediction|')
    plt.title('In-silico mutagenesis')
    plt.tight_layout()
    plt.savefig(args.output, dpi=180)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='results/dna_cnn.pt')
    p.add_argument('--sequence', required=True)
    p.add_argument('--output', default='results/in_silico_mutagenesis.png')
    main(p.parse_args())
