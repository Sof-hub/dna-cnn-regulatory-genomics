import numpy as np

BASES = 'ACGT'
BASE_TO_IDX = {b:i for i,b in enumerate(BASES)}

# JASPAR MA0138.2 (REST, human), transcribed from the published PFM.
# Rows: A, C, G, T. Used only to generate a controlled synthetic benchmark.
REST_PFM = np.array([
[211,58,76,1452,34,122,1575,3,35,913,219,39,20,1406,13,1574,42,205,366,213,179],
[174,269,1366,30,44,978,7,1586,1481,201,375,7,5,112,1280,9,13,1007,31,933,1114],
[366,146,50,94,1516,323,12,12,20,162,124,1551,1577,34,233,7,1530,183,688,319,37],
[840,1124,105,26,10,182,12,5,70,329,886,7,2,51,74,10,9,198,507,125,260]
], dtype=float)


def pwm_from_pfm(pfm, pseudocount=0.5):
    probs = (pfm + pseudocount) / (pfm.sum(axis=0, keepdims=True) + pseudocount * 4)
    return probs


def sample_motif(rng, pwm):
    return ''.join(rng.choice(list(BASES), p=pwm[:, j]) for j in range(pwm.shape[1]))


def random_background(rng, length):
    return ''.join(rng.choice(list(BASES), size=length))


def one_hot(sequences):
    x = np.zeros((len(sequences), 4, len(sequences[0])), dtype=np.float32)
    for i, seq in enumerate(sequences):
        for j, base in enumerate(seq):
            x[i, BASE_TO_IDX[base], j] = 1.0
    return x


def make_dataset(n=12000, seq_len=200, positive_fraction=0.5, seed=42):
    rng = np.random.default_rng(seed)
    pwm = pwm_from_pfm(REST_PFM)
    sequences, labels = [], []
    motif_len = pwm.shape[1]
    for _ in range(n):
        y = int(rng.random() < positive_fraction)
        seq = random_background(rng, seq_len)
        if y:
            pos = int(rng.integers(0, seq_len - motif_len + 1))
            motif = sample_motif(rng, pwm)
            seq = seq[:pos] + motif + seq[pos + motif_len:]
        sequences.append(seq)
        labels.append(y)
    return sequences, np.asarray(labels, dtype=np.float32)
