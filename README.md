# 🧬 Predict 2D Chromatin Contact Maps (Hi-C) from 1D Open Chromatin (ATAC-seq)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Executive Summary

High-throughput chromosome conformation capture (**Hi-C**) provides deep insights into 3D genome architecture, including Topologically Associating Domains (TADs) and enhancer-promoter loops. However, generating Hi-C datasets remains exceptionally expensive and computationally intensive. 

In contrast, **ATAC-seq** (Assay for Transposase-Accessible Chromatin) measures 1D chromatin accessibility at a fraction of the cost. 

**Core Objective:** Build a lightweight, end-to-end deep learning framework utilizing **Dilated Convolutional Neural Networks (CNNs)** to predict high-resolution 2D Hi-C contact maps using only cheap 1D ATAC-seq accessibility signals.

---

## 📊 Dataset & Biological Context

Data was ingested directly from matched human cell line experiments via the **ENCODE Consortium**:

* **Input Data (1D Feature):** ATAC-seq BigWig signal (`ENCFF667MDI`) representing open vs. closed chromatin accessibility.
* **Target Data (2D Matrix):** Hi-C contact matrix (`ENCFF216ZNY`) capturing spatial contact frequency.
* **Genome Partitioning:**
  * **Training Chromosomes:** `chr1` and `chr2` (899 non-sparse, clean windows).
  * **Validation/Test Chromosome (Held-Out):** `chr21` (64 non-sparse, clean windows). Holding out `chr21` ensures strict evaluation of model generalization across unseen genomic loci.

---

## 🛠️ Data Preprocessing & Feature Engineering

1. **Resolution Binning:** Binned both 1D signal and 2D Hi-C matrices into **10 kb resolution bins**.
2. **Window Slicing:** Sliced chromosomes into non-overlapping **100-bin ($1\text{ Mb}$) contiguous windows**, producing matching pairs of $[1, 100]$ input vectors and $[100, 100]$ output matrices.
3. **Sparsity Filtering:** Automatically excluded unmapped, centromeric, and low-coverage regions ($>75\%$ zeros in Hi-C targets) to eliminate background noise.
4. **Dynamic Scaling:** Applied a logarithmic transformation $\log1p(Y) = \ln(1 + Y)$ to raw Hi-C counts to compress dynamic range near the primary diagonal ($i = j$) and stabilize training.

---

## 📐 Network Architecture

The repository implements `DilatedGenomicPredictor` (~42,625 parameters), structured as follows:

```text
1D ATAC-seq Input [Batch, 1, 100]
       │
  1D Dilated Conv Encoder (Dilation Rates d = 1, 2, 4, 8)
       │
  Pairwise Outer Product Spatial Broadcasting (1D ➔ 2D Expansion)
       │
  Concatenate Relative Spatial Distance Matrix Prior |i - j|
       │
  2D Dilated Conv Decoder
  Predited 2D Log(Hi-C) 
Matrix [Batch, 100, 100]
