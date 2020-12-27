# Code for Cross-dataset Training

[[Cross-dataset Training for Class Increasing Object Detection](http://arxiv.org/abs/2001.04621)]

## Example
For human face cross training, we have two datasets, one is labeled for face, the other is labeled for human. We want to get a model which can detect human and face. (human label as 1, face label as 2)
The negative target from human dataset, we relabel the negative target as 3; the negative target from face dataset, we relabel the negative target as 4. We need to ingore the confused negative samples.

python cross_focal_loss.py


## Citation

If you find this Model & Software useful in your research we would kindly ask you to cite:

```bibtex
@misc{2001.04621,
Author = {Yongqiang Yao and Yan Wang and Yu Guo and Jiaojiao Lin and Hongwei Qin and Junjie Yan},
Title = {Cross-dataset Training for Class Increasing Object Detection},
Year = {2020},
Eprint = {arXiv:2001.04621},
}
```
