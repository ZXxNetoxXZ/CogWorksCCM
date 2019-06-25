""" Keras is an interface that wraps multiple frameworks like tensorflow, cntk, or theano and is Official high level api of 
    tensorflow. Similar to PyTorch, KERAS makes getting started with DeepLearning easier. """

import numpy as np


def convert_kernel(kernel, dim_ordering):
    """ Converts a kernel matrix (Numpy array) from Theano format to
        TensorFlow format, and vice-versa.

        Parameters
        ----------
        kernel : numpy.ndarray
            Theano or TensorFlow-style convolution kernel
        dim_ordering : str
            Theano-style: 'th'
            TensorFlow-style: 'tf'

        If Theano-style kernel, the kernel shape is: (C, F, H, W) or (C, F, H, W, D)
        If TensorFlow-style kernel, the kernel shape is: (H, W, C, F) or (H, W, D, C, F)

        Reverse the height (H), width (W), and (if present) depth (D) axes.
        C & F axes should not be changed.

        C : number of channels of data
        F : number of convolution filters
        H : height dimension
        W : width dimension
        D : depth dimension """
    new_kernel = np.copy(kernel)
    if kernel.ndim == 4:
        # conv 2d
        # TH kernel shape: (depth, input_depth, rows, cols)
        # TF kernel shape: (rows, cols, input_depth, depth)
        if dim_ordering == 'th':
            w = kernel.shape[2]
            h = kernel.shape[3]
            for i in range(w):
                for j in range(h):
                    new_kernel[:, :, i, j] = kernel[:, :, w - i - 1, h - j - 1]
        elif dim_ordering == 'tf':
            w = kernel.shape[0]
            h = kernel.shape[1]
            for i in range(w):
                for j in range(h):
                    new_kernel[i, j, :, :] = kernel[w - i - 1, h - j - 1, :, :]
        else:
            raise ValueError('Invalid dim_ordering:', dim_ordering)
    elif kernel.ndim == 5:
        # conv 3d
        # TH kernel shape: (out_depth, input_depth, kernel_dim1, kernel_dim2, kernel_dim3)
        # TF kernel shape: (kernel_dim1, kernel_dim2, kernel_dim3, input_depth, out_depth)
        if dim_ordering == 'th':
            w = kernel.shape[2]
            h = kernel.shape[3]
            z = kernel.shape[4]
            for i in range(w):
                for j in range(h):
                    for k in range(z):
                        new_kernel[:, :, i, j, k] = kernel[:, :,
                                                           w - i - 1,
                                                           h - j - 1,
                                                           z - k - 1]
        elif dim_ordering == 'tf':
            w = kernel.shape[0]
            h = kernel.shape[1]
            z = kernel.shape[2]
            for i in range(w):
                for j in range(h):
                    for k in range(z):
                        new_kernel[i, j, k, :, :] = kernel[w - i - 1,
                                                           h - j - 1,
                                                           z - k - 1,
                                                           :, :]
    return new_kernel