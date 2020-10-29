# TigeCMN
Learning Temporal Interaction Graph Embedding via Coupled Memory Networks (WWW-2020)

This is a Pytorch implementation of the proposed TigeCMN algorithm, which learns node representations in temporal interaction graphs. Specifically, TigeCMN consists of two modules, i.e., user-specific module and item-specific module, which is made up of two key operations, e.g., writing and reading operations.

## Requirements
* python3.6
* pytorch
* sklearn
* numpy
* tqdm

## Basic Usage

### Input Data
For node classification, each dataset contains 3 files: user iteraction sequence file, item interaction sequence file and node label file.
```
1. Delta_Time_DBLP_User.dat: this file has n+1 lines.
The first line has the following format:
user_num@::item_num
The next n lines are as follows: (each node per line ordered by user node id):
user_id@::item_id@::time_interval@::contents

2. Delta_Time_DBLP_Item.dat: this file has n+1 lines.
The first line has the following format:
user_num@::item_num
The next n lines are as follows: (each node per line ordered by item node id):
item_id@::user_id@::time_interval@::contents

3. DBLP_Label.dat: this file has n lines, each line represents a node and its class label.
node_1 label_1
node_2 label_2
...
```

### Run
To run TigeCMN, just execute the following command for node classification task:
```
python main_classification.py
```

## Citing
If you find TigeCMN useful for your research, please consider citing the following paper:
```
@inproceedings{zhang2020learning,
  title={Learning Temporal Interaction Graph Embedding via Coupled Memory Networks},
  author={Zhang, Zhen and Bu, Jiajun and Ester, Martin and Zhang, Jianfeng and Yao, Chengwei and Li, Zhao and Wang, Can},
  booktitle={Proceedings of The Web Conference 2020},
  pages={3049--3055},
  year={2020}
}
```