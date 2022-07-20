# RS_Repo

This repository is used to keep track of the code I have written for the recommendation system.

## Algorithms (keep updating)
**NCF** —— [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031) WWW, 2017

## Datasets
Since most of the code in this library exists to solve my homework, most of the datasets are specified by the teacher.

**Tianchi_Train_Test** 
This dataset was intercepted from the Tianchi competition. Its training set consists of three columns: user ID, item ID and corresponding score. And testing set consists of two columns: User ID and item ID.

## How to start

main.py is kept in src/workflow/, and just run it. In addition, You can adjust the parameters corresponding to the algorithm in the src/config/ folder. It‘s recommended to use wandb's sweep to search for the best parameters.