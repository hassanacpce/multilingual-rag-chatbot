"""
Generates a small synthetic sample in the EXACT schema of the real Kaggle
arXiv metadata snapshot (arxiv-metadata-oai-snapshot.json), for testing the
pipeline end-to-end without needing the real ~4GB file / Kaggle credentials.

Real schema (one JSON object per line, despite the .json extension):
  id, submitter, authors, title, comments, journal-ref, doi, report-no,
  categories, license, abstract, versions, update_date, authors_parsed
"""
import json
import random

random.seed(42)

TOPICS = [
    ("cs.LG", "Machine Learning", [
        ("Efficient Fine-Tuning of Large Language Models via Low-Rank Adaptation",
         "We study parameter-efficient fine-tuning methods for large pretrained language models. "
         "Full fine-tuning of billion-parameter models is computationally expensive and memory intensive. "
         "We propose a low-rank adaptation technique that freezes the pretrained weights and injects "
         "trainable rank-decomposition matrices into each layer of the transformer architecture, "
         "drastically reducing the number of trainable parameters. Experiments on natural language "
         "understanding and generation benchmarks show performance comparable to full fine-tuning "
         "while using orders of magnitude fewer trainable parameters and less GPU memory."),
        ("Graph Neural Networks for Molecular Property Prediction",
         "Predicting molecular properties from graph-structured representations of chemical compounds "
         "is a central problem in computational chemistry. We present a graph neural network architecture "
         "that combines message passing with attention mechanisms to capture both local atomic environments "
         "and long-range dependencies within a molecule. Our model is evaluated on several benchmark "
         "datasets for quantum property prediction and toxicity classification, achieving state-of-the-art "
         "accuracy while remaining computationally efficient."),
        ("Reinforcement Learning with Human Feedback for Text Summarization",
         "We investigate the use of reinforcement learning from human feedback to align text summarization "
         "models with human preferences. A reward model is trained on pairwise human comparisons of "
         "candidate summaries, and a policy is optimized against this reward model using proximal policy "
         "optimization. We find that RLHF-tuned summarizers produce outputs that are preferred by human "
         "evaluators over standard supervised fine-tuning baselines, particularly on coherence and factual "
         "consistency metrics."),
    ]),
    ("cs.CL", "Computation and Language", [
        ("Attention Is All You Need for Long-Document Summarization",
         "Long documents pose a challenge for transformer-based summarization models due to the quadratic "
         "cost of self-attention with respect to sequence length. We propose a sparse attention mechanism "
         "that restricts attention computation to a fixed window combined with a small number of global "
         "attention tokens, enabling the model to process documents of several thousand tokens. Evaluation "
         "on scientific paper and news summarization benchmarks demonstrates improved ROUGE scores and "
         "substantially reduced memory usage compared to full self-attention baselines."),
        ("Few-Shot Named Entity Recognition via Prompt-Based Learning",
         "Named entity recognition typically requires large amounts of labeled training data. We reformulate "
         "the entity recognition task as a cloze-style prompting problem, allowing a pretrained masked "
         "language model to be adapted to new entity types with only a handful of labeled examples. "
         "Our prompt-based approach outperforms conventional fine-tuning in the few-shot regime across "
         "multiple domains, including biomedical and legal text."),
        ("Retrieval-Augmented Generation for Open-Domain Question Answering",
         "We combine a dense passage retriever with a sequence-to-sequence generator to answer open-domain "
         "questions by conditioning generation on retrieved evidence passages. The retriever and generator "
         "are trained jointly end-to-end, allowing the retrieval component to adapt to the information needs "
         "of the downstream generation task. Our method achieves strong performance on open-domain QA "
         "benchmarks while providing interpretable evidence for each generated answer."),
    ]),
    ("cs.CV", "Computer Vision", [
        ("Self-Supervised Contrastive Learning for Medical Image Segmentation",
         "Labeled medical imaging data is scarce and expensive to obtain. We propose a self-supervised "
         "contrastive pretraining scheme that learns useful visual representations from unlabeled medical "
         "scans, which are then fine-tuned for downstream segmentation tasks using a small number of labeled "
         "examples. Our approach substantially improves segmentation accuracy on MRI and CT datasets when "
         "labeled data is limited, compared to training from randomly initialized weights."),
        ("Vision Transformers for Real-Time Object Detection",
         "Convolutional neural networks have long dominated real-time object detection due to their "
         "computational efficiency. We adapt the vision transformer architecture for object detection by "
         "introducing a lightweight patch embedding scheme and a hierarchical feature pyramid, enabling "
         "real-time inference on standard hardware. Our detector achieves competitive accuracy with "
         "convolutional baselines while offering better scaling with training data size."),
    ]),
    ("cs.AI", "Artificial Intelligence", [
        ("Explainable Multi-Agent Reinforcement Learning for Traffic Signal Control",
         "Coordinating traffic signals across an urban road network is a challenging multi-agent control "
         "problem. We develop a multi-agent reinforcement learning framework in which each intersection is "
         "controlled by an independent agent that learns to cooperate with neighboring agents through "
         "communication. We further introduce an attention-based explanation module that surfaces which "
         "neighboring intersections most influenced a given signal decision, improving the interpretability "
         "of the learned policies for traffic engineers."),
    ]),
    ("cs.DS", "Data Structures and Algorithms", [
        ("Approximate Nearest Neighbor Search in High-Dimensional Spaces",
         "Exact nearest neighbor search becomes computationally intractable as dimensionality grows. "
         "We present a graph-based approximate nearest neighbor algorithm that constructs a navigable "
         "small-world graph over the dataset, enabling logarithmic-time search with high recall. "
         "We provide theoretical guarantees on search complexity and demonstrate empirically that our "
         "method outperforms tree-based and hashing-based approaches on billion-scale vector datasets."),
    ]),
    ("cs.CR", "Cryptography and Security", [
        ("Differential Privacy Guarantees for Federated Learning Systems",
         "Federated learning enables model training across decentralized devices without directly sharing "
         "raw data, but gradient updates can still leak sensitive information. We analyze the privacy "
         "guarantees of adding calibrated noise to gradient updates under the differential privacy framework, "
         "and propose an adaptive clipping strategy that improves the privacy-utility tradeoff compared to "
         "fixed clipping thresholds, evaluated across several federated benchmark tasks."),
    ]),
    ("math.NA", "Numerical Analysis", [
        ("Neural Network Solvers for Partial Differential Equations",
         "We explore the use of deep neural networks as function approximators for solving partial "
         "differential equations, using the residual of the governing equations as a training loss. "
         "This physics-informed approach avoids the need for labeled solution data and generalizes across "
         "a family of boundary conditions once trained, though we identify limitations in accuracy for "
         "stiff and high-frequency solution components."),
    ]),
]

AUTHOR_POOL = [
    ("Chen", "Wei"), ("Kumar", "Priya"), ("Smith", "John"), ("Garcia", "Maria"),
    ("Wang", "Li"), ("Johnson", "Sarah"), ("Müller", "Anna"), ("Nakamura", "Yuki"),
    ("Petrov", "Ivan"), ("Silva", "Carlos"), ("Okafor", "Chidi"), ("Dubois", "Claire"),
]


def make_entry(idx, category, category_name, title, abstract):
    n_authors = random.randint(1, 3)
    authors_parsed = random.sample(AUTHOR_POOL, n_authors)
    authors_str = ", ".join(f"{f} {l}" for l, f in authors_parsed)
    arxiv_id = f"25{random.randint(1,9):02d}.{idx:05d}"
    return {
        "id": arxiv_id,
        "submitter": f"{authors_parsed[0][1]} {authors_parsed[0][0]}",
        "authors": authors_str,
        "title": title,
        "comments": f"{random.randint(6,20)} pages, {random.randint(2,10)} figures",
        "journal-ref": None,
        "doi": None,
        "report-no": None,
        "categories": category,
        "license": "http://creativecommons.org/licenses/by/4.0/",
        "abstract": " " + abstract + " ",
        "versions": [{"version": "v1", "created": "Mon, 12 Jan 2026 00:00:00 GMT"}],
        "update_date": "2026-01-12",
        "authors_parsed": [[l, f, ""] for l, f in authors_parsed],
    }


def main():
    idx = 1
    entries = []
    for category, category_name, papers in TOPICS:
        for title, abstract in papers:
            entries.append(make_entry(idx, category, category_name, title, abstract))
            idx += 1

    with open("arxiv-metadata-oai-snapshot-sample.json", "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    print(f"Wrote {len(entries)} synthetic sample entries -> arxiv-metadata-oai-snapshot-sample.json")


if __name__ == "__main__":
    main()
