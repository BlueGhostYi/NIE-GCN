# NIE-GCN
This is the PyTorch implementation for the paper:
> Zhang, Yi, et al. "[NIE-GCN: Neighbor Item Embedding-Aware Graph Convolutional Network for Recommendation.](https://ieeexplore.ieee.org/abstract/document/10413999)" IEEE Transactions on Systems, Man, and Cybernetics: Systems (2024).

## Environment
```
python == 3.8.18
pytorch == 2.1.0 (cuda:12.1)
scipy == 1.10.1
numpy == 1.24.3
```
## Settings
* yelp2018: batch_size=1024, beta=1.0, l2=1e-4

* amazon-book: batch_size=2048, beta=0.8, l2=1e-5

* movielens: batch_size=1024, beta=0.9, l2=1e-4, agg='cat'

  
## Citation
If you find this work is helpful to your research, please consider citing our paper:
```
@article{zhang2024nie,
  title={NIE-GCN: Neighbor Item Embedding-Aware Graph Convolutional Network for Recommendation},
  author={Zhang, Yi and Zhang, Yiwen and Yan, Dengcheng and He, Qiang and Yang, Yun},
  journal={IEEE Transactions on Systems, Man, and Cybernetics: Systems},
  year={2024},
  volume={54},
  number={5},
  pages={2810-2821},
  publisher={IEEE}
}
```
