# Adversarial Training Over Long-Tailed Distribution [[pdf](https://arxiv.org/abs/2307.10205)]

Introduction: We explore the challenges and limitations of adversarial training on a long-tailed dataset. It is not to address the long-tail problem itself. Instead, we study how to improve adversarial training when the training data is imbalanced.


## Requirements
1. pytorch >= 1.9.0
2. torchvision
3. numpy
4. tqdm
5. mmcv

## Adversarial Training

``python train.py --arch [resnet, wrn]
--dataset [cifar10, cifar100] --imb [imbalanced ratio] --ext [existing ratio]
--save [the name you want to save your model]
--exp [experiment name]``




