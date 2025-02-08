# CCGCN: Complex Composition Graph Convolutional Neural Networks for Temporal Knowledge Graph Reasoning

This repository provides the implementation of the paper "CCGCN: Complex Composition Graph Convolutional Neural Networks for Temporal Knowledge Graph Reasoning". The goal of this project is to develop a novel neural network architecture that can effectively reason over temporal knowledge graphs by leveraging complex composition and graph convolutional networks. This repository includes the code for training and evaluating the proposed model on benchmark datasets.

## Dependency

- Python 3.9
- CUDA version 11.7

## How to run the project?

- Step 1: Set up the environments

```
cd ccgcn

conda create -n ccgcn python=3.9

conda activate ccgn

pip install -r requirements.txt

```

- Step 2: Run the training process,

Example

```
python main.py -model "CCGCN" -dataset "icews14" -run_id test -lr 0.002 -gcn_inp_dim 50 -gcn_hid_dim 100 -emb_dim 25 -tim_dim 75 -inp_drop 0.2 -hid_drop 0.1 -gcn_drop 0.3 -fin_drop 0.4 -save_each 1 -ne 1 -bsize 1

```

*Note*: Config other parameters if you need

## Authors

Loc Tran, Bac Le and Thanh Le

## References

This implementation is based on [DE-SimplE](https://github.com/BorealisAI/de-simple)and [CompGCN](https://github.com/toooooodo/CompGCN-DGL).We appreciate their awesome work.