# Learning Regulatory DNA Motifs with a 1D CNN

A small, reproducible deep-learning project demonstrating how a convolutional neural network can learn sequence motifs from DNA and how its predictions can be interrogated with **in-silico mutagenesis**.

> **Portfolio note:** this repository is a controlled proof-of-concept, not a biological claim. The labels are generated from a JASPAR transcription-factor profile embedded into random DNA backgrounds. The goal is to demonstrate a complete and reproducible deep-learning workflow before moving to real genomic assays.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![Task](https://img.shields.io/badge/Task-DNA%20classification-2ea44f)
![Status](https://img.shields.io/badge/Status-portfolio%20project-informational)

## 1. Biological question

Can a neural network learn a transcription-factor binding pattern directly from DNA sequence?

The project uses the human **REST** transcription-factor profile **JASPAR MA0138.2** as a controlled motif source. JASPAR is an open-access database of curated transcription-factor binding profiles represented as position-frequency matrices (PFMs). The current JASPAR release also contains deep-learning models and discovered profiles.

The model is trained to distinguish:

- **positive sequence:** random DNA containing a sampled REST-like motif;
- **negative sequence:** random DNA without the implanted motif.

This is deliberately simpler than a real regulatory genomics task. It gives a clean test of whether the CNN can recover sequence patterns and whether interpretation methods identify the relevant positions.

## 2. Why this project matters for genomics

The same basic idea appears in real regulatory genomics:

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

## 3. What I implemented

### Input

Each DNA sequence has length 200 bp and is encoded with one-hot encoding:

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

The first convolution can be interpreted as a bank of learned sequence detectors. The network is not given the motif as a rule; it adjusts its parameters from labelled examples.

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
3. calculate how parameters contributed to the error;
4. update the parameters;
5. repeat.

The loss is `BCEWithLogitsLoss`, which combines a binary classification objective with the sigmoid operation in a numerically stable implementation.

### Evaluation

I report more than accuracy because classification performance should not be reduced to a single thresholded number. The project reports:

- ROC-AUC
- Average Precision / PR-AUC
- accuracy
- precision
- recall
- F1

## 4. Example result

A CPU run with 4,000 synthetic sequences, 5 epochs and seed 42 produced:

| Metric | Test value |
| --- | ---: |
| ROC-AUC | 0.9984 |
| Average Precision | 0.9985 |
| Accuracy | 0.9733 |
| Precision | 0.9965 |
| Recall | 0.9505 |
| F1 | 0.9730 |

These numbers are included to demonstrate that the repository was actually executed. A sample mutagenesis visualization is available in [`assets/in_silico_mutagenesis.png`](assets/in_silico_mutagenesis.png). They should not be interpreted as evidence of biological performance because the benchmark is synthetic and intentionally controlled.

## 5. Interpretation: in-silico mutagenesis

After training, the model can be interrogated by changing one nucleotide at a time.

For a reference sequence:

```text
prediction(reference) = 0.82
```

we create mutants such as:

```text
A -> C
A -> G
A -> T
```

and recompute the prediction. A large change indicates that the corresponding position strongly affects the model's output.

This is **not** proof that a nucleotide is biologically causal. It tells us which sequence positions are important to the trained model. In real genomics, such model-based evidence can be combined with experimental evidence to generate hypotheses.

## 6. Reproducibility

### Install

```bash
git clone <YOUR-REPOSITORY-URL>
cd dna-cnn-portfolio
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
# .venv\\Scripts\\activate    # Windows
pip install -r requirements.txt
```

### Run tests

```bash
pytest
```

### Train the model

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

The script reports the mutations producing the largest changes in the model prediction and saves a position-wise effect plot.

## 7. Project structure

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
└── results/
```

## 8. Limitations

This project should not be presented as a biological predictor.

Important limitations:

1. The dataset is synthetic.
2. The positive class is defined by motif implantation, so the task is easier than real TF-binding prediction.
3. There is no cell-type-specific chromatin context.
4. The model does not predict gene expression.
5. The project does not model L1 retrotransposition or insertion effects directly.
6. In-silico mutagenesis explains model sensitivity, not biological causality.
7. A real genomic study would require careful train/test splitting to avoid sequence homology and genomic leakage.

## 9. Next steps

### Step 1 — Real TF-binding data

Replace synthetic labels with experimental binding/chromatin data from ENCODE or another public resource.

### Step 2 — Genomic sequences

Use real human genomic regions obtained from a reference genome, e.g. through Ensembl or a downloaded reference assembly.

### Step 3 — Stronger baselines

Compare:

- logistic regression on k-mer features;
- shallow CNN;
- deeper CNN;
- CNN + attention;
- transformer-based sequence model.

### Step 4 — Better experimental design

Use chromosome- or region-aware splits when appropriate and explicitly check for homologous sequences and leakage.

### Step 5 — L1-oriented formulation

Define a real target such as the change in transcription associated with an L1 insertion and integrate local sequence with functional-genomics context.

### Step 6 — Interpretability

Compare:

- in-silico mutagenesis;
- saliency / gradient-based attribution;
- SHAP where computationally appropriate;
- motif enrichment against known TF profiles.

## 10. References

1. Alipanahi, B., Delong, A., Weirauch, M. T., & Frey, B. J. (2015). Predicting the sequence specificities of DNA- and RNA-binding proteins by deep learning. *Nature Biotechnology*, 33, 831–838. <https://doi.org/10.1038/nbt.3300>

2. Zhou, J., & Troyanskaya, O. G. (2015). Predicting effects of noncoding variants with deep learning-based sequence model. *Nature Methods*, 12, 931–934. <https://doi.org/10.1038/nmeth.3547>

3. Avsec, Ž., Agarwal, V., Visentin, D., et al. (2021). Effective gene expression prediction from sequence by integrating long-range interactions. *Nature Methods*, 18, 1196–1203. <https://doi.org/10.1038/s41592-021-01252-x>

4. Ovek, D. B., Rauluseviciute, I., Aronsen, D. R., et al. (2026). JASPAR 2026: expansion of transcription factor binding profiles and integration of deep learning models. *Nucleic Acids Research*, 54(D1), D184–D193. <https://doi.org/10.1093/nar/gkaf1209>

5. ENCODE Consortium. ENCODE Portal: public genomic and functional-genomics data. <https://www.encodeproject.org/>

6. Ensembl. Ensembl REST API, sequence endpoints. <https://rest.ensembl.org/>

7. Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.

8. MIT Introduction to Deep Learning (6.S191). Software labs and course materials. <https://github.com/MITDeepLearning/introtodeeplearning>
