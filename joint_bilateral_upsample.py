'''
Joint bilateral upsampling class
'''

import numpy as np
import itertools as it

def clamp(min_value: float, max_value: float, x: np.ndarray) -> np.ndarray:
    return np.maximum(min_value, np.minimum(max_value, x))

def loop_Nlinear_interpolation(array: np.ndarray, y_coord: np.ndarray, x_coord: np.ndarray, r_coord: np.ndarray, g_coord: np.ndarray, b_coord: np.ndarray) -> np.ndarray:
    # Index order: y, x, r, g, b
    y_size, x_size, r_size, g_size, b_size = array.shape

    # Flatten, (y_size x x_size x r_size x g_size x b_size)
    array_flat = array.reshape((-1, ))

    # Left and right indices of the interpolation
    def get_both_indices(size, coord):
        left_index = clamp(0, size - 1, coord.astype(np.int32))
        right_index = clamp(0, size - 1, left_index + 1)
        return left_index, right_index
    
    y_index, yy_index = get_both_indices(y_size, y_coord) # (h, w)
    x_index, xx_index = get_both_indices(x_size, x_coord) # (h, w)
    r_index, rr_index = get_both_indices(r_size, r_coord) # (h, w)
    g_index, gg_index = get_both_indices(g_size, g_coord) # (h, w)
    b_index, bb_index = get_both_indices(b_size, b_coord) # (h, w)

    left_indices = [y_index, x_index, r_index, g_index, b_index]
    right_indices = [yy_index, xx_index, rr_index, gg_index, bb_index]

    # Coordinates
    def coord_transform(y_idx, x_idx, r_idx, g_idx, b_idx):
        return ((((y_idx * x_size + x_idx) * r_size + r_idx) * g_size + g_idx) * b_size + b_idx).reshape((-1))

    y_alpha = (y_coord - y_index).reshape((-1, )) # (h x w, )
    x_alpha = (x_coord - x_index).reshape((-1, )) # (h x w, )
    r_alpha = (r_coord - r_index).reshape((-1, )) # (h x w, )
    g_alpha = (g_coord - g_index).reshape((-1, )) # (h x w, )
    b_alpha = (b_coord - b_index).reshape((-1, )) # (h x w, )
    alphas = [y_alpha, x_alpha, r_alpha, g_alpha, b_alpha] # (5, h x w)

    interpolation = np.zeros_like(y_alpha)
    for perm in it.product(range(2), repeat=5):
        alpha_prod = np.ones_like(y_alpha)
        idx = []
        for i in range(len(perm)):
            if perm[i] == 1:
                alpha_prod *= (1. - alphas[i])
                idx.append(left_indices[i])
            else:
                alpha_prod *= alphas[i]
                idx.append(right_indices[i])

        interpolation += alpha_prod * array_flat[coord_transform(*idx)]

    return interpolation

class JointBilateralUpsampling:
    '''
    Joint bilateral upsampling
    '''
    def __init__(self, height: int, width: int, space_sigma: float=16, range_sigma: float=.25, padding_xy: int=2, padding_z: int=2) -> None:
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
        self.space_sigma, self.range_sigma = space_sigma, range_sigma
        self.padding_xy, self.padding_z = padding_xy, padding_z

        self.data = None

    def init(self, features: np.ndarray) -> None:
        # `features` should be three-channel and channel-last
        assert features.ndim == 3 and features.shape[-1] == 3
        
        # Decomposing into r, g, and b channels
        r, g, b = features[..., 0], features[..., 1], features[..., 2]
        r_min, r_max = r.min(), r.max()
        g_min, g_max = g.min(), g.max()
        b_min, b_max = b.min(), b.max()
        r_delta, g_delta, b_delta = r_max - r_min, g_max - g_min, b_max - b_min

        # Space coordinates, shape (h, w), dtype int
        yy, xx = np.mgrid[:self.height, :self.width]
        # Range coordinates, shape (h, w), dtype float
        rr, gg, bb = r - r_min, g - g_min, b - b_min

        # High-dim bilateral grid
        # Shape of `data`, scala, dtype size_type
        self.small_height = self.size_type((self.height - 1) / self.space_sigma) + 1 + 2 * self.padding_xy 
        self.small_width = self.size_type((self.width - 1) / self.space_sigma) + 1 + 2 * self.padding_xy
        self.small_rdepth = self.size_type(r_delta / self.range_sigma) + 1 + 2 * self.padding_z 
        self.small_gdepth = self.size_type(g_delta / self.range_sigma) + 1 + 2 * self.padding_z 
        self.small_bdepth = self.size_type(b_delta / self.range_sigma) + 1 + 2 * self.padding_z
        # Declare `data`
        self.data_shape = (self.small_height, self.small_width, self.small_rdepth, self.small_gdepth, self.small_bdepth)
        self.data = np.zeros(self.data_shape, dtype=np.float32) # For each channel
        self.data_flat = self.data.reshape((-1, )) # view of data

        # Generating splat coordinates
        # Space coordinates, shape (h, w)
        splat_yy = (yy / self.space_sigma + .5).astype(np.int32) + self.padding_xy
        splat_xx = (xx / self.space_sigma + .5).astype(np.int32) + self.padding_xy
        # Range coordinates, shape (h, w)
        splat_rr = (rr / self.range_sigma + .5).astype(np.int32) + self.padding_z
        splat_gg = (gg / self.range_sigma + .5).astype(np.int32) + self.padding_z
        splat_bb = (bb / self.range_sigma + .5).astype(np.int32) + self.padding_z
        # Splat coordinates, shape (h x w, )
        self.splat_coords = (((splat_yy * self.small_width + splat_xx) * self.small_rdepth + splat_rr) * self.small_gdepth + splat_gg) * self.small_bdepth + splat_bb
        self.splat_coords = self.splat_coords.reshape((-1)) # (h x w, )

        # Generating slice coordinates
        # Space coordinates, shape (h, w)
        slice_yy = yy.astype(np.float32) / self.space_sigma + self.padding_xy
        slice_xx = xx.astype(np.float32) / self.space_sigma + self.padding_xy
        # Range coordinates, shape (h, w)
        slice_rr = rr / self.range_sigma + self.padding_z
        slice_gg = gg / self.range_sigma + self.padding_z
        slice_bb = bb / self.range_sigma + self.padding_z
        # Slice coordinates
        self.slice_coords = [slice_yy, slice_xx, slice_rr, slice_gg, slice_bb]

        # Interpolation
        self.interpolation = np.zeros((self.height * self.width, ), dtype=np.float32)
        
        # Left and right indices of the interpolation
        def get_both_indices(size, coord):
            left_index = clamp(0, size - 1, coord.astype(np.int32))
            right_index = clamp(0, size - 1, left_index + 1)
            return left_index, right_index
        
        y_index, yy_index = get_both_indices(self.small_height, slice_yy) # (h, w)
        x_index, xx_index = get_both_indices(self.small_width, slice_xx) # (h, w)
        r_index, rr_index = get_both_indices(self.small_rdepth, slice_rr) # (h, w)
        g_index, gg_index = get_both_indices(self.small_gdepth, slice_gg) # (h, w)
        b_index, bb_index = get_both_indices(self.small_bdepth, slice_bb) # (h, w)

        self.left_indices = [y_index, x_index, r_index, g_index, b_index]
        self.right_indices = [yy_index, xx_index, rr_index, gg_index, bb_index]

        y_alpha = (slice_yy - y_index).reshape((-1, )) # (h x w, )
        x_alpha = (slice_xx - x_index).reshape((-1, )) # (h x w, )
        r_alpha = (slice_rr - r_index).reshape((-1, )) # (h x w, )
        g_alpha = (slice_gg - g_index).reshape((-1, )) # (h x w, )
        b_alpha = (slice_bb - b_index).reshape((-1, )) # (h x w, )
        self.alphas = [y_alpha, x_alpha, r_alpha, g_alpha, b_alpha] # (5, h x w)

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
        def coord_transform(y_idx, x_idx, r_idx, g_idx, b_idx):
            return ((((y_idx * self.small_width + x_idx) * self.small_rdepth + r_idx) * self.small_gdepth + g_idx) * self.small_bdepth + b_idx).reshape((-1))
        
        # Initialize interpolation
        self.interpolation.fill(0)

        for perm in it.product(range(2), repeat=5):
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
