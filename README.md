# Amrita Vishwa Vidyapeetham

## Amrita School of Artificial Intelligence

---

# MedBot: Prompt-Engineered Biomedical Question Answering System

---

## Team Members

* Jignesh Sudheer — CB.SC.U4AIE24222 — B.Sc AI
* Sharavn RM — CB.SC.U4AIE24253 — B.Sc AI
* Bhadhresh R — CB.SC.U4AIE24208 — B.Sc AI
* Gautham T — CB.SC.U4AIE24264 — B.Sc AI

---

## Table of Contents

1. Abstract
2. Introduction
3. Methodology
4. Dataset
5. Implementation
6. Results
7. Conclusion
8. References

---

## Abstract

This project presents MedBot, a biomedical question-answering system based on BioGPT. The system integrates prompt engineering and domain-specific fine-tuning on the BioASQ dataset to generate structured and context-aware medical responses. The model achieves a BLEU-2 score of 0.42 and demonstrates improved performance compared to baseline approaches.

---

## Introduction

Biomedical question answering requires both domain-specific understanding and natural language generation capability. Traditional encoder-based models are limited in generating free-form responses.

MedBot addresses this limitation by using a decoder-only transformer (BioGPT) combined with prompt engineering. The system processes medical queries and generates responses similar to expert explanations.

---

## Methodology

### Problem Formulation

Given a medical query:

$$
Q \rightarrow A
$$

where

* $Q$ is the input medical question
* $A$ is the generated biomedical answer

---

### Model Architecture

The system uses a decoder-only transformer:

* Token Embedding
* Positional Encoding
* Multi-head Self Attention
* Transformer Decoder Blocks
* Linear Projection + Softmax

---

### Prompt Engineering

Prompt templates are dynamically created based on question type:

* Definition
* Symptoms
* Treatment
* Causes
* Prevention

This improves structure and relevance of generated answers.

---

### Training Objective

The model is trained using cross-entropy loss:

$$
\mathcal{L} = - \sum \log P(w_i | w_1, ..., w_{i-1})
$$

---

## Dataset

* Dataset: BioASQ
* Format: Question-Answer pairs
* Total samples: ~12,000

Data is converted into:

```
Question: <query> Answer: <response>
```

---

## Implementation

### 1. Dataset Processing

Implemented in `data_processor.py` 

* Loads BioASQ JSON
* Converts into text format
* Tokenizes using BioGPT tokenizer
* Outputs input_ids and attention_mask

---

### 2. Model

Implemented in `model.py` 

Key features:

* Uses pretrained BioGPT
* Detects question type
* Generates structured prompts
* Uses beam search and sampling
* Applies response cleaning

---

### 3. Training Pipeline

Implemented in `train.py` 

* Optimizer: AdamW
* Learning Rate: $2 \times 10^{-5}$
* Epochs: 5
* Batch size: 2
* Loss: Cross-Entropy

Model checkpoints are saved after each epoch.

---

## Results

### 1. Training Performance

Fig. 1: Training vs Validation Loss

The loss decreases steadily across epochs, showing convergence.

---

### 2. BLEU Score

* Final BLEU-2 Score: 0.42
* Baseline: 0.33

Fig. 2: BLEU Score across Question Types

* Highest: Yes/No
* Lowest: Summary

---

### 3. BLEU Progression

Fig. 3: BLEU Score vs Epoch

Score improves from 0.31 to 0.42.

---

### 4. Token Analysis

Fig. 4: Answer Length Distribution

* Most responses: 35–55 tokens

Fig. 5: Input Length Distribution

* Most inputs: 80–100 tokens

---

### 5. Confidence Analysis

Fig. 6: Confidence vs Error

* Higher confidence → lower error

---

### 6. Attention Analysis

Fig. 7: Token Attention Heatmap

* Model focuses on key biomedical tokens

---

## Conclusion

MedBot demonstrates the effectiveness of combining BioGPT with prompt engineering for biomedical question answering. The system produces structured, context-aware responses and shows improved performance across evaluation metrics.

This approach can be extended to real-world applications such as medical education and research assistance.

---

## References

1. BioGPT Paper
2. BioASQ Dataset
3. Transformer Architecture
4. BLEU Score Paper
5. Hugging Face Transformers

---
