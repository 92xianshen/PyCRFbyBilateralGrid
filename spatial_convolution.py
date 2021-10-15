import numpy as np
import PIL.Image as Image
import cv2
import itertools as it

def clamp(min_value: float, max_value: float, x: np.ndarray) -> np.ndarray:
    return np.maximum(min_value, np.minimum(max_value, x))

class SpatialConvolution:
    '''
    Spatial convolution
    '''
    def __init__(self, height: int, width: int, space_sigma: float=16, padding_xy: int=2) -> None:
        '''
        Initializer

        Args:
            base: edge or guidance image, shape (h, w, 3)
            space_sigma: sigma_s, float
            range_sigma: sigma_r, float

        Returns:
            None
        '''
        # Datatype cast
        self.size_type = int
        # Index order: y --> height, x --> width, z --> depth
        self.height, self.width = height, width
        self.space_sigma = space_sigma
        self.padding_xy = padding_xy

        self.data = None

    def init(self) -> None:
        # Space coordinates, shape (h, w), dtype int
        yy, xx = np.mgrid[:self.height, :self.width]

        # Spatial grid
        # Shape of `data`, scala, dtype size_type
        self.small_height = self.size_type((self.height - 1) / self.space_sigma) + 1 + 2 * self.padding_xy 
        self.small_width = self.size_type((self.width - 1) / self.space_sigma) + 1 + 2 * self.padding_xy
        # Declare `data`
        self.data_shape = (self.small_height, self.small_width)
        self.data = np.zeros(self.data_shape, dtype=np.float32) # For each channel
        self.data_flat = self.data.reshape((-1, )) # view of data

        # Generating splat coordinates
        # Space coordinates, shape (h, w)
        splat_yy = (yy / self.space_sigma + .5).astype(np.int32) + self.padding_xy
        splat_xx = (xx / self.space_sigma + .5).astype(np.int32) + self.padding_xy
        # Splat coordinates, shape (h x w, )
        self.splat_coords = splat_yy * self.small_width + splat_xx
        self.splat_coords = self.splat_coords.reshape((-1)) # (h x w, )

        # Generating slice coordinates
        # Space coordinates, shape (h, w)
        slice_yy = yy.astype(np.float32) / self.space_sigma + self.padding_xy
        slice_xx = xx.astype(np.float32) / self.space_sigma + self.padding_xy
        # Slice coordinates
        self.slice_coords = [slice_yy, slice_xx]

        # Interpolation
        self.interpolation = np.zeros((self.height * self.width, ), dtype=np.float32)
        
        # Left and right indices of the interpolation
        def get_both_indices(size, coord):
            left_index = clamp(0, size - 1, coord.astype(np.int32))
            right_index = clamp(0, size - 1, left_index + 1)
            return left_index, right_index
        
        y_index, yy_index = get_both_indices(self.small_height, slice_yy) # (h, w)
        x_index, xx_index = get_both_indices(self.small_width, slice_xx) # (h, w)

        self.left_indices = [y_index, x_index]
        self.right_indices = [yy_index, xx_index]

        y_alpha = (slice_yy - y_index).reshape((-1, )) # (h x w, )
        x_alpha = (slice_xx - x_index).reshape((-1, )) # (h x w, )
        self.alphas = [y_alpha, x_alpha] # (5, h x w)

        self.alpha_prod = np.ones((self.height * self.width, ), dtype=np.float32)

    def convn(self, n_iter: int=2):
        buffer = np.zeros_like(self.data)
        perm = list(range(1, self.data.ndim)) + [0] # [1, ..., ndim - 1, 0] 

        for _ in range(n_iter):
            buffer, self.data = self.data, buffer

            for dim in range(self.data.ndim):
                self.data[1:-1] = (buffer[:-2] + buffer[2:] + 2. * buffer[1:-1]) / 4.
                self.data = np.transpose(self.data, perm)
                buffer = np.transpose(buffer, perm)

        del buffer

    def loop_Nlinear_interpolation(self) -> np.ndarray:
        # Coordinates
        def coord_transform(y_idx, x_idx):
            return (y_idx * self.small_width + x_idx).reshape((-1))
        
        # Initialize interpolation
        self.interpolation.fill(0)

        for perm in it.product(range(2), repeat=2):
            self.alpha_prod.fill(1)
            idx = []
            
            for i in range(len(perm)):
                if perm[i] == 1:
                    self.alpha_prod *= (1. - self.alphas[i])
                    idx.append(self.left_indices[i])
                else:
                    self.alpha_prod *= self.alphas[i]
                    idx.append(self.right_indices[i])

            self.interpolation += self.alpha_prod * self.data_flat[coord_transform(*idx)]

    def compute(self, inp: np.ndarray, out: np.ndarray):
        assert inp.shape == out.shape
        _, _, n_channels = inp.shape[:3]

        # For each channel
        for ch in range(n_channels):
            inp_ch = inp[..., ch]
            out_ch = out[..., ch]

            # ==== Splat ====
            # Reshape, shape (h x w)
            inp_flat = inp_ch.reshape((-1, ))
            # Splatting
            self.data_flat[:] = np.bincount(self.splat_coords, minlength=self.data_flat.shape[0], weights=inp_flat)

            # ==== Blur ====
            # 5D convolution
            self.convn(n_iter=2)

            # ==== Slice ====
            # Interpolation
            self.loop_Nlinear_interpolation()

            # Get result0
            out_ch[:] = self.interpolation.reshape((self.height, self.width))

def spatial_convolve(data: np.ndarray, n_iter: int=2) -> None:
    buffer = np.zeros_like(data)

    for _ in range(n_iter):
        buffer, data = data, buffer

        # For dim y
        data[1:-1, :] = (buffer[:-2, :] + buffer[2:, :] + 2. * buffer[1:-1, :]) / 4.
        # For dim x
        data[:, 1:-1] = (buffer[:, :-2] + buffer[:, 2:] + 2. * buffer[:, 1:-1]) / 4.

    del buffer