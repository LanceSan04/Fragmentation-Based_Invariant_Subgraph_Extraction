


## [AAAI 2025] Subgraph Aggregation for Out-of-Distribution Generalization on Graphs

Official PyTorch implementation of the paper [Subgraph Aggregation for Out-of-Distribution Generalization on Graphs](https://arxiv.org/abs/2410.22228)(SuGAr) in AAAI 2025.

## Pipeline

<p align="center"><img src="./data/arch.png"></p>


## Instructions

### Installation and data preparation
run the following command to install environment:
```sh
pip install -r requirement.txt
```

plus the [DrugOOD](https://github.com/tencent-ailab/DrugOOD) benchmark repo and [config](https://github.com/LFhase/CIGA/tree/main/configs).

The data used in the paper can be obtained following these [instructions](./dataset_gen/README.md).

### Running example
Train 
```sh
python main.py -c_in 'feat' -c_rep 'feat'  --seed '[1,2,3,4,5,6,7,8,9,10]' --num_layers 3 --dataset 'SUMotif' --bias 0.6 --r 0.25 --contrast 2 --spu_coe 0 --model 'gcn' --dropout 0 --epoch 40 --init 1 --penalty 1 --sample_rate 95 --save_model
```

Ensemble
```sh
python ens.py -c_in 'feat' -c_rep 'feat'   --num_layers 3 --dataset 'SUMotif' --bias 0.6 --r 0.25 --contrast 2 --spu_coe 0 --model 'gcn' --dropout 0 --init 1 --penalty 1 --sample_rate 95 --divp
```
## Misc
If you find our paper and repo useful, please cite our paper:
```bibtex
@article{Liu_2025_sugar, 
title={Subgraph Aggregation for Out-of-Distribution Generalization on Graphs}, volume={39}, 
url={https://ojs.aaai.org/index.php/AAAI/article/view/34065}, 
DOI={10.1609/aaai.v39i18.34065},  
number={18}, 
journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
author={Liu, Bowen and Li, Haoyang and Wang, Shuning and Nie, Shuo and Zhang, Shanghang}, 
year={2025}, 
month={Apr.}, 
pages={18763-18771} 
}
```

We would like to acknowledge the contribution from [CIGA](https://github.com/LFhase/CIGA) to the base codes.