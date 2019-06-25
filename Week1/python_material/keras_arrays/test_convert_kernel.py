def test_convert_kernel(func):
    """ Tests kera's kernel-convert for 4D and 5D arrays, with Theano
        and TensorFlow styled kernels.

        Prints whether or not each test passed.

        Parameters
        ----------
        func : Callable[[numpy.ndarray], numpy.ndarray]
            The function to to be tested. """
    import numpy as np
    from itertools import product
    from pathlib import Path
    dims = [4, 5]
    modes = ['tf', 'th']
    with np.load(Path(".") / "kernels.npz") as f:
        for dim, mode in product(dims, modes):
            k = f["{mode}{dim}d".format(mode=mode, dim=dim)]
            kf = f["{mode}{dim}df".format(mode=mode, dim=dim)]
            compare = (func(k, mode) == kf)
            t = "passed" if np.all(compare) else "failed"
            print("kenel-mode: {mode}; dimensionality: {dim}D: {tf}".format(mode=mode,
                                                                            dim=dim,
                                                                            tf=t))
