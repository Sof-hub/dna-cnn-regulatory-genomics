# Learning Regulatory DNA Motifs with a 1D CNN

A small, reproducible deep-learning project demonstrating how a convolutional neural network can learn regulatory DNA motifs from sequence and how its predictions can be interrogated with **in-silico mutagenesis**.

> **Portfolio note:** this repository is a controlled proof-of-concept, not a biological claim. The labels are generated from a JASPAR transcription-factor profile embedded into random DNA backgrounds. The goal is to demonstrate a complete and reproducible deep-learning workflow before moving to real genomic assays.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![Task](https://img.shields.io/badge/Task-DNA%20classification-2ea44f)
![Status](https://img.shields.io/badge/Status-portfolio%20project-informational)

## 1. Biological question

Can a neural network learn a transcription-factor binding pattern directly from DNA sequence?

The project uses the human **REST** transcription-factor profile **JASPAR MA0138.2** as a controlled motif source. JASPAR is an open-access database of curated transcription-factor binding profiles represented as position-frequency matrices (PFMs).

The model is trained to distinguish:

* **positive sequence:** random DNA containing a sampled REST-like motif;
* **negative sequence:** random DNA without the implanted motif.

This is deliberately simpler than a real regulatory genomics task. It provides a controlled test of whether a CNN can recover a sequence pattern and whether interpretation methods identify the relevant positions.

## 2. Why this project matters for genomics

The same basic workflow appears in real regulatory genomics:

```text
DNA sequence
     ↓
sequence representation
     ↓
CNN / other sequence model
     ↓
prediction of a regulatory phenotype
     ↓
interpretation
     ↓
biological hypothesis
```

Deep learning has been used to infer DNA/RNA-binding specificities from experimental data and to predict regulatory effects of noncoding sequence. DeepSEA, for example, learned sequence features associated with chromatin effects and could evaluate single-nucleotide changes. Enformer extended sequence-based prediction to long-range genomic context and gene-expression-related tracks.

This project intentionally focuses on the first part of that workflow: learning a local sequence motif from a controlled synthetic benchmark.

## 3. What I implemented

### Input

Each DNA sequence has length 200 bp and is encoded using one-hot encoding:

```text
A -> [1,0,0,0]
C -> [0,1,0,0]
G -> [0,0,1,0]
T -> [0,0,0,1]
```

For a batch, PyTorch receives a tensor of shape:

```text
[B, 4, L]
```

where `B` is the batch size and `L=200` is sequence length.

### Model

```text
[ B, 4, 200 ]
       ↓
Conv1D: 64 filters, kernel 21
       ↓
ReLU
       ↓
MaxPool
       ↓
Conv1D: 128 filters, kernel 7
       ↓
ReLU
       ↓
Global max pooling
       ↓
Linear layer
       ↓
logit
       ↓
sigmoid
       ↓
P(REST-like motif)
```

The first convolution acts as a bank of learned sequence detectors. The network is not given a hard-coded classification rule. Its parameters are learned from labelled examples.

### Training

The training loop follows the standard supervised-learning pattern:

```python
optimizer.zero_grad()
logits = model(x)
loss = criterion(logits, y)
loss.backward()
optimizer.step()
```

In simple terms:

1. make a prediction;
2. measure the error;
3. calculate how the parameters contributed to the error;
4. update the parameters;
5. repeat.

The loss is `BCEWithLogitsLoss`, which combines the binary classification objective with the sigmoid operation in a numerically stable implementation.

### Evaluation

The project reports:

* ROC-AUC
* Average Precision
* accuracy
* precision
* recall
* F1

Accuracy alone is not sufficient to characterize a binary classifier, so multiple complementary metrics are reported.

## 4. Results

The main reproducible run used:

| Parameter           |  Value |
| ------------------- | -----: |
| Synthetic sequences | 12,000 |
| Sequence length     | 200 bp |
| Epochs              |      8 |
| Batch size          |    128 |
| Learning rate       |  0.001 |
| Random seed         |     42 |
| Device              |    CPU |

The resulting test metrics were:

| Metric            | Test value |
| ----------------- | ---------: |
| ROC-AUC           | **0.9995** |
| Average Precision | **0.9995** |
| Accuracy          | **0.9900** |
| Precision         | **0.9890** |
| Recall            | **0.9912** |
| F1                | **0.9901** |

These results show that the CNN can very effectively distinguish the two classes in this controlled synthetic benchmark.

They should **not** be interpreted as evidence of biological performance. The positive class is explicitly constructed by inserting a REST-like motif into otherwise random DNA, making the task substantially easier than real transcription-factor binding prediction.

The training configuration is saved in:

```text
results/config.json
```

and the evaluation metrics are saved in:

```text
results/metrics.json
```

## 5. Interpretation: in-silico mutagenesis

After training, the model can be interrogated by changing one nucleotide at a time.

For each position, the original nucleotide is replaced by each alternative base:

```text
A -> C
A -> G
A -> T
```

The model is then evaluated again.

The current implementation measures the effect using the change in the model's **raw output logit**:

```text
delta_logit = mutant_logit - reference_logit
```

This is preferable to comparing probabilities when the model is highly confident, because the sigmoid can saturate near 0 or 1 and hide substantial differences in the underlying model output.

A large absolute `delta_logit` indicates that the corresponding mutation strongly affects the model's prediction.

This is **not** proof that a nucleotide is biologically causal. It identifies positions to which the trained model is sensitive. In real genomics, such model-based evidence would need to be combined with experimental evidence.

## 6. Example interpretation result

For one positive 200-bp sequence generated with the same synthetic dataset procedure, the implanted REST-like motif was:

```text
TCCAGCACCACGGACAGCTCC
```

located at positions:

```text
84-104
```

The reference sequence produced:

```text
reference_logit = 24.538935
reference_probability = 1.000000
```

The strongest mutations were concentrated inside the implanted motif:

| Position | Mutation | Delta logit |
| -------: | :------: | ----------: |
|       99 |    A>G   |      -9.176 |
|       90 |    A>T   |      -9.058 |
|       95 |    G>T   |      -9.007 |
|       90 |    A>C   |      -8.981 |
|       99 |    A>T   |      -8.903 |
|       91 |    C>T   |      -8.479 |
|      100 |    G>T   |      -8.318 |
|       88 |    G>T   |      -8.292 |
|       90 |    A>G   |      -8.171 |
|       91 |    C>A   |      -8.153 |

The most sensitive positions are therefore concentrated within the region containing the implanted motif.

This provides a useful sanity check: the model's strongest sequence sensitivities align with the local sequence pattern that defines the positive class.

A written interpretation of this analysis is available in:

```text
results/analysis.txt
```

The mutagenesis script also produces a position-wise effect plot:

```text
results/in_silico_mutagenesis.png
```

## 7. Reproducibility

### Install

```bash
git clone <REPOSITORY-URL>
cd dna-cnn-regulatory-genomics
python -m venv .venv
```

Activate the environment:

```bash
# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Run tests

```bash
pytest
```

### Train the model

The main run can be reproduced with:

```bash
python src/train.py --n-samples 12000 --epochs 8 --seed 42 --out-dir results
```

The script saves:

```text
results/
├── dna_cnn.pt
├── metrics.json
└── config.json
```

### Run in-silico mutagenesis

After training, provide one 200-bp sequence:

```bash
python src/interpret.py \
  --model results/dna_cnn.pt \
  --sequence YOUR_200BP_SEQUENCE \
  --output results/in_silico_mutagenesis.png
```

On Windows PowerShell, the same command can be written on one line:

```powershell
python src/interpret.py --model results/dna_cnn.pt --sequence 200BP_SEQUENCE --output results/in_silico_mutagenesis.png
```

The script reports the mutations producing the largest changes in the model logit and saves a position-wise effect plot.

## 8. Project structure

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── interpret.py
│   └── utils.py
├── tests/
│   └── test_project.py
├── notebooks/
├── results/
│   ├── config.json
│   ├── metrics.json
│   ├── dna_cnn.pt
│   ├── analysis.txt
│   ├── example_positive_sequence.txt
│   ├── example_sequence.txt
│   └── in_silico_mutagenesis.png
```

Generated model files may be excluded from version control depending on the repository configuration.

## 9. Limitations

This project should **not** be presented as a biological predictor.

Important limitations:

1. The dataset is synthetic.
2. The positive class is defined by motif implantation, so the task is easier than real TF-binding prediction.
3. There is no cell-type-specific chromatin context.
4. The model does not predict gene expression.
5. The project does not model L1 retrotransposition or insertion effects directly.
6. In-silico mutagenesis measures model sensitivity, not biological causality.
7. A real genomic study would require careful train/test splitting to avoid sequence homology and genomic leakage.
8. The very high performance is expected in this controlled benchmark and should not be extrapolated to real genomic datasets.

## 10. Next steps

### Step 1 — Real TF-binding data

Replace synthetic labels with experimental binding or chromatin data from ENCODE or another public resource.

### Step 2 — Genomic sequences

Use real human genomic regions obtained from a reference genome, for example through Ensembl or a downloaded reference assembly.

### Step 3 — Stronger baselines

Compare:

* logistic regression on k-mer features;
* shallow CNN;
* deeper CNN;
* CNN + attention;
* transformer-based sequence model.

### Step 4 — Better experimental design

Use chromosome- or region-aware splits when appropriate and explicitly check for homologous sequences and leakage.

### Step 5 — L1-oriented formulation

Define a real target such as the change in transcription associated with an L1 insertion and integrate local sequence with functional-genomics context.

### Step 6 — Interpretability

Compare:

* in-silico mutagenesis;
* saliency / gradient-based attribution;
* SHAP where computationally appropriate;
* motif enrichment against known TF profiles.

## 11. References

1. Alipanahi, B., Delong, A., Weirauch, M. T., & Frey, B. J. (2015). Predicting the sequence specificities of DNA- and RNA-binding proteins by deep learning. *Nature Biotechnology*, 33, 831–838. <https://doi.org/10.1038/nbt.3300>

2. Zhou, J., & Troyanskaya, O. G. (2015). Predicting effects of noncoding variants with deep learning-based sequence model. *Nature Methods*, 12, 931–934. <https://doi.org/10.1038/nmeth.3547>

3. Avsec, Ž., Agarwal, V., Visentin, D., et al. (2021). Effective gene expression prediction from sequence by integrating long-range interactions. *Nature Methods*, 18, 1196–1203. <https://doi.org/10.1038/s41592-021-01252-x>

4. Ovek, D. B., Rauluseviciute, I., Aronsen, D. R., et al. (2026). JASPAR 2026: expansion of transcription factor binding profiles and integration of deep learning models. *Nucleic Acids Research*, 54(D1), D184–D193. <https://doi.org/10.1093/nar/gkaf1209>

5. ENCODE Consortium. ENCODE Portal: public genomic and functional-genomics data. <https://www.encodeproject.org/>

6. Ensembl. Ensembl REST API, sequence endpoints. <https://rest.ensembl.org/>

7. Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.

8. MIT Introduction to Deep Learning (6.S191). Software labs and course materials. <https://github.com/MITDeepLearning/introtodeeplearning>
