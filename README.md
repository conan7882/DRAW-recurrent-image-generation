# DRAW-tensorflow
TensorFlow implementation of [DRAW: A Recurrent Neural Network For Image Generation](https://arxiv.org/pdf/1502.04623.pdf) (ICML 15).
 - The the model from the paper:
 ![ram](figs/model.png)
 - It is an RNN version of variational autoencoder.
 - During generation, at each time step, the code z is sampled from the prior p(z) and fed into decoder. Then the decoder modifies part of the canvas through writer operation. At last step, the canvas C_T is used to compute p(x | z_(1:T)).
 - During training, at each time step, the input image and error image is encoded through read operation and encoder RNN. Then the output of encoder is used to estimate the posterior of code z.
 - Attention mechanism can be appiled to `read` and `write` operations, which utilize an array of estimated 2D Gaussian filters at each time step. 
 
 
