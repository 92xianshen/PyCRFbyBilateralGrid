import numpy as np
import PIL.Image as Image
import cv2

from spatial_convolution import SpatialConvolution

# ==== Test grayscale joint bilateral upsampling ====

im = Image.open('lena.jpg')
im = np.asarray(im, dtype='float32') / 255.
height, width, n_channels = im.shape[:3]

spconv = SpatialConvolution(height, width, space_sigma=16.)
spconv.init()

inp = im

result, weight = np.zeros_like(inp, dtype=np.float32), np.zeros((height, width, 1), dtype=np.float32)
all_ones = np.ones_like(weight, dtype=np.float32)

spconv.compute(inp, result)
spconv.compute(all_ones, weight)

result = result / (weight + 1e-5)

cv2.imshow('im', im[..., ::-1])
cv2.imshow('result', result[..., ::-1])
cv2.waitKey()