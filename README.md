# variational-conv-dequantization
PyTorch implementation for variational dequantization using convolutions.

Module that performs variational dequantization similarly to Flow++ (https://arxiv.org/abs/1902.00275)

Used when dealing with spatially dependent quantized embeddings, i.e mu and var are obtained from a feature vector that is the result of a convolution operation with kernel_size > 1

The feature vector for z_{B x H x W} is obtained by performing a convolution around z_{B x H x W}, then a MLP extracts mu_{B x H x W}, respectively var_{B x H x W}
