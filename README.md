# Random tensor stuff

This repository contains pytorch implementations of randomised tensor regression layer (RTRL) and randomised tensor contraction layer (RTCL) and the notebooks with experiments with them.

## Folders
- *models* - in this folder you can find the implementations of all the layers.
- *notebooks* - here are all the experiments, complementary notebooks, training pipelines and a subfolder with already trained models.

## Short description
In both layers implemented here the tensor-train decompposition is used to store the weights of regression and contraction layers. It is done in the same way as in [Novikov et al](https://arxiv.org/pdf/1509.06569.pdf) and [Garipov et al](https://arxiv.org/pdf/1611.03214.pdf) respectively.
After that the dropout on the rank of a tensor in tensor-train is applyed as it is done in [Kolbeinsson et al](https://arxiv.org/pdf/1902.10758.pdf).
Storing weights in tensor decomposition reduces the number of learnable parameters of the layer and adding dropout increases robustness of it to adversarial attacks.
