# Reproduction of Deep auto-encoder based clustering by Song et al.

The reference for this paper is :

```bibtex
@article{Song2014DeepAB,
  title={Deep auto-encoder based clustering},
  author={Chunfeng Song and Yongzhen Huang and Feng Liu and Zhenyu Wang and Liang Wang},
  journal={Intell. Data Anal.},
  year={2014},
  volume={18},
  pages={S65-S76}
}
```

This is a TensorFlow reimplementation of the deep auto-encoder method for clustering presented by Song et al. in 2014.
The method is tested on MNIST and compared to the results reported in the paper.
The running implementation is in [the Jupyter notebook](mnist_clustering.ipynb), the implementation of the loss function is in [this file](song_loss.py), and the auto-encoder is found [here](song_network.py).

Sadly, for now, the results are not reproduced.
PR are welcome to try and reproduce the results.