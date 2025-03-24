# LoRA: Low-Rank Adaptation from Scratch

This repo contains my personal implementation and understanding of [LoRA (Low-Rank Adaptation of Large Language Models)](https://arxiv.org/abs/2106.09685), built entirely in PyTorch.

## Contents

- `lora-implementation.ipynb` – Implementation of LoRA from scratch
- `lora-from-scratch.md` – My personal write-up: theory, intuition, and code walkthrough

## Quick Overview

LoRA allows you to fine-tune large language models efficiently by freezing the pre-trained weights and injecting small, trainable low-rank matrices into select layers (typically attention).

It works by approximating the full weight update using:

$$ W = W_0 + \Delta W $$

Instead of updating the full matrix, we only learn `A` and `B`, keeping everything else frozen.

## Full Write-up

👉 [Read the full explanation here](./lora-from-scratch.md)

---