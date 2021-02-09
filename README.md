# Structured Sparsity Inducing Adaptive Optimizers for Deep Learning

This is the repository for the paper

> Tristan Deleu, Yoshua Bengio, *Structured Sparsity Inducing Adaptive Optimizers for Deep Learning* [[ArXiv](https://arxiv.org/abs/2102.03869)]

This repository contains:
 - The weighted and unweighted proximal operators for the l1/l2 and group MCP penalties
 - A modification of [AdamW](https://arxiv.org/abs/1711.05101) from Hugging Face's [transformers](https://huggingface.co/transformers/) library to include a proximal step, compatible with the structured sparsity inducing penalties in this repository.
 - The definition of the groups (channel-wise & row-wise) for some Deep Learning architectures (VGG, Resnet, BERT).
