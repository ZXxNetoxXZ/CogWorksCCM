""" These are all proposed method from stackoverflow  for converting an image
    array of shape (H, W, 3), whose channels are ordered as BGR, to RGB...
    they are all way too complicated."""


def meth1(im):
    from PIL import Image
    b, g, r = im.split()
    return Image.merge("RGB", (r, g, b))


def meth2(im):
    import numpy as np
    from PIL import Image
    data = np.asarray(im)
    return Image.fromarray(np.roll(data, 1, axis=-1))


def meth3(path):
    import cv2
    srcBGR = cv2.imread(path)
    return cv2.cvtColor(srcBGR, cv2.COLOR_BGR2RGB)


def meth4(bgr_image):
    import numpy
    from PIL import Image
    bgr_image_array = numpy.asarray(bgr_image)
    B, G, R = bgr_image_array.T
    rgb_image_array = numpy.array((R, G, B)).T
    return Image.fromarray(rgb_image_array, mode='RGB')