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
        ref_logit = model(torch.from_numpy(original).to(device)).item()
        ref_prob = torch.sigmoid(
            torch.tensor(ref_logit)
        ).item()

    scores = []

    for pos in range(len(sequence)):
        for base in bases:
            if base == sequence[pos]:
                continue

            mutant = mutate_sequence(sequence, pos, base)

            with torch.no_grad():
                mutant_input = torch.from_numpy(
                    one_hot([mutant])
                ).to(device)

                mutant_logit = model(mutant_input).item()

            delta = mutant_logit - ref_logit
            scores.append((pos, sequence[pos], base, delta))

    return ref_logit, ref_prob, scores


def main(args):
    device = torch.device('cpu')

    model = DNAConvNet().to(device)
    model.load_state_dict(
        torch.load(args.model, map_location=device)
    )
    model.eval()

    ref_logit, ref_prob, scores = in_silico_mutagenesis(
        model,
        args.sequence,
        device
    )

    print(f'reference_logit={ref_logit:.6f}')
    print(f'reference_probability={ref_prob:.6f}')

    best = sorted(
        scores,
        key=lambda x: abs(x[3]),
        reverse=True
    )[:20]

    for row in best:
        print(
            f'position={row[0]:03d} '
            f'{row[1]}>{row[2]} '
            f'delta_logit={row[3]:+.6f}'
        )

    pos_effect = np.zeros(len(args.sequence))

    for pos, _, _, delta in scores:
        pos_effect[pos] = max(
            pos_effect[pos],
            abs(delta)
        )

    plt.figure(figsize=(12, 3))
    plt.plot(pos_effect)
    plt.xlabel('Sequence position')
    plt.ylabel('Max |delta logit|')
    plt.title('In-silico mutagenesis')
    plt.tight_layout()
    plt.savefig(args.output, dpi=180)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument(
        '--model',
        default='results/dna_cnn.pt'
    )
    p.add_argument(
        '--sequence',
        required=True
    )
    p.add_argument(
        '--output',
        default='results/in_silico_mutagenesis.png'
    )

    main(p.parse_args())