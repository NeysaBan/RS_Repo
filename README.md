# RS_Repo

This repository is used to keep track of the code I have written for the recommendation system.

## π€ Algorithms (keep updating)
**NCF** ββ [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031) WWW, 2017

**DeepcoNN** ββ [Joint Deep Modeling of Users and Items Using Reviews for Recommendation](https://dl.acm.org/doi/abs/10.1145/3018661.3018665) WSDM, 2017

**SASRec** ββ [Self-Attentive Sequential Recommendation](https://ieeexplore.ieee.org/abstract/document/8594844) IEEE, 2018

## β€οΈβπ₯ Datasets
Since most of the code in this library exists to solve my homework, most of the datasets are specified by the teacher. And you can download them [there](https://drive.google.com/drive/folders/1AJH5DGcLM6X82CLdEZPdpdUR0QorULEw?usp=sharing)

**Tianchi_Train_Test** 
This dataset is applied to NCF. Itwas intercepted from the Tianchi competition. Its training dataset consists of three columns: user ID, item ID and corresponding score. And testing dataset consists of two columns: User ID and item ID.

**Digital_Train_Test**
The dataset is applied to DeepCoNN. It contains training dataset and testing dataset. Training dataset consists of four columns: user ID, item ID, review and rating. Testing dataset consists of two columns: user ID and item ID.

**ml-1m**
This dataset is applied to SASRec. It is modified from [MovieLens](https://grouplens.org/datasets/movielens/) dataset generated by grouplens. It cosists only two columns: user ID and item ID.

## π©π»βπ» How to start

main.py is kept in src/workflow/, and just run it. In addition, You can adjust the parameters corresponding to the algorithm in the src/config/ folder. Itβs recommended to use wandb's sweep to search for the best parameters.