<img width="543" height="153" alt="image" src="https://github.com/user-attachments/assets/479d9ba6-0f4b-4f99-8576-7e60e6df769a" />


## Amrita School of Artificial Intelligence

---

# MedBot: Prompt-Engineered Biomedical Question Answering System

---

## Team Members

* Jignesh Sudheer — CB.SC.U4AIE24222 
* Sharavn RM — CB.SC.U4AIE24253 
* Bhadhresh R — CB.SC.U4AIE24208 
* Gautham T — CB.SC.U4AIE24264 

---

## Table of Contents

1. Abstract
2. Introduction
3. Methodology
4. Mathematical Formulation
5. Dataset
6. Implementation
7. Results and Inference
8. Conclusion
9. References

---

## Abstract

This project presents MedBot, a biomedical question-answering system built using BioGPT and enhanced through prompt engineering. The system is trained on the BioASQ dataset to generate medically relevant and structured responses. The model achieves a BLEU-2 score of 0.42, demonstrating improved performance over baseline approaches. The results show that combining domain-specific pretraining with structured prompts enhances both accuracy and coherence.

---

## Introduction

Biomedical question answering requires understanding complex domain-specific language and generating meaningful responses. Traditional encoder-based models are limited in free-form text generation.

MedBot addresses this by using a decoder-only transformer model and prompt engineering to generate structured and context-aware biomedical answers.

---

## Methodology

### Problem Definition

Given a medical query:

$$
Q \rightarrow A
$$

where

* $Q$ is the input question
* $A$ is the generated answer

---

### Model Architecture

The system uses a decoder-only transformer (BioGPT), which generates text autoregressively.

---

### Prompt Engineering

Prompt templates are designed based on question type to guide the model’s output structure and improve relevance.

---

## Mathematical Formulation

### 1. Language Modeling Objective

The model learns the probability of a sequence of tokens:

$$
P(W) = \prod_{i=1}^{N} P(w_i \mid w_1, w_2, ..., w_{i-1})
$$

This represents autoregressive generation where each token depends on previous tokens.

---

### 2. Training Loss (Cross-Entropy)

The objective is to minimize cross-entropy loss:

$$
\mathcal{L} = - \sum_{i=1}^{N} \log P(w_i \mid w_1, ..., w_{i-1})
$$

This ensures the generated sequence closely matches the ground truth.

---

### 3. Self-Attention Mechanism

The core of the transformer is self-attention:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

where

* $Q$ = Query
* $K$ = Key
* $V$ = Value
* $d_k$ = dimension scaling factor

This allows the model to focus on relevant tokens.

---

### 4. Multi-Head Attention

Multiple attention heads are used:

$$
\text{MultiHead}(Q,K,V) = \text{Concat}(head_1,...,head_h)W^O
$$

Each head captures different contextual relationships.

---

### 5. Transformer Layer

Each decoder layer consists of:

$$
\text{LayerNorm}(x + \text{Attention}(x))
$$

$$
\text{LayerNorm}(x + \text{FeedForward}(x))
$$

This ensures stable learning and deep representation.

---

### 6. Softmax Output

The probability distribution over vocabulary:

$$
P(w_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

---

### 7. BLEU Score

Evaluation metric:

$$
BLEU = BP \cdot \exp \left( \sum_{n=1}^{N} w_n \log p_n \right)
$$

This measures similarity between generated and reference text.

---

### 8. Perplexity

Model uncertainty:

$$
PP(W) = \sqrt[N]{\prod_{i=1}^{N} \frac{1}{P(w_i \mid w_1,...,w_{i-1})}}
$$

Lower perplexity indicates better predictions.

---

### 9. Correlation (Confidence vs Error)

$$
r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}}
$$

This measures reliability of model predictions.

---

## Dataset

The model is trained on the BioASQ dataset consisting of biomedical question-answer pairs. The dataset includes multiple question types such as definition, factoid, treatment, and yes/no.

---

## Implementation

* Dataset processing using tokenization
* BioGPT pretrained model for generation
* Prompt-based input formatting
* Training using AdamW optimizer

---

## Results and Inference

### Training Behavior

The decreasing training and validation loss indicates stable learning and good generalization. The model does not show signs of severe overfitting.

---

### BLEU Score Analysis

A BLEU-2 score of 0.42 indicates strong alignment with reference answers. Structured question types achieve higher accuracy due to predictable response patterns.

---

### Learning Trend

The steady increase in BLEU score across epochs confirms effective fine-tuning and learning progression.

---

### Response Quality

The generated responses are coherent, medically relevant, and well-structured. The model maintains an optimal response length, balancing detail and clarity.

---

### Token Behavior

The model adapts to varying input lengths and produces consistent output lengths similar to expert responses.

---

### Reliability

The inverse relationship between confidence and error demonstrates that the model is well-calibrated and reliable.

---

### Attention Insight

The attention mechanism focuses on critical biomedical terms, enabling accurate and context-aware responses.

---

### Overall Inference

The integration of transformer architecture and prompt engineering results in:

* Improved semantic understanding
* Better response structuring
* Enhanced accuracy across question types

---

## Conclusion

MedBot successfully demonstrates the application of transformer-based models in biomedical question answering. The use of prompt engineering significantly enhances response quality.

Future improvements can include larger datasets, model scaling, and integration with real-time clinical systems.

---

## References

1. BioGPT Paper
2. BioASQ Dataset
3. Attention is All You Need
4. BLEU Score Paper
5. Hugging Face Transformers

---
