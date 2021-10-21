'''
High-dimensional filter
spatial high-dim filter if `features` is None
bilateral high-dim filter if `features` is np.ndarray
'''

import numpy as np
import itertools as it

def clamp(min_value: float, max_value: float, x: np.ndarray) -> np.ndarray:
    return np.maximum(min_value, np.minimum(max_value, x))

# Method to get left and right indices of slice interpolation
def get_both_indices(size: np.int32, coord: np.ndarray):
    left_index = clamp(0, size - 1, coord.astype(np.int32))
    right_index = clamp(0, size - 1, left_index + 1)
    return left_index, right_index

class HighDimFilter:
    '''
    High-dimensional filter
    '''
    def __init__(self, is_bilateral: bool, height: int, width: int, space_sigma: float=16, range_sigma: float=.25, padding_xy: int=2, padding_z: int=2) -> None:
        '''
        Initializer

        Args:
            height: height of image to be filtered, int
            width: width of image to be filtered, int
            space_sigma: sigma_s, float
            range_sigma: sigma_r, float
            padding_xy: number of pixel for padding along y and x, int
            padding_z: number of pixel for padding along z, int

        Returns:
            None
        '''
        # Is bilateral?
        self.is_bilateral = is_bilateral
        # Datatype cast
        self.size_type = int
        # Index order: y --> height, x --> width, z --> depth
        self.height, self.width = height, width
        self.size = height * width
        self.space_sigma, self.range_sigma = space_sigma, range_sigma
        self.padding_xy, self.padding_z = padding_xy, padding_z

    def init(self, features: np.ndarray=None) -> None:
        # Initialize a spatial high-dim filter if `features` is None; otherwise initialize a bilateral one and `features` should be three-channel and channel-last
        if self.is_bilateral:
            assert features.ndim == 3 and features.shape[-1] == 3

        # Height and width of grid, scala, dtype size_type
        small_height = self.size_type((self.height - 1) / self.space_sigma) + 1 + 2 * self.padding_xy 
        small_width = self.size_type((self.width - 1) / self.space_sigma) + 1 + 2 * self.padding_xy

        # Space coordinates, shape (h, w), dtype int
        yy, xx = np.mgrid[:self.height, :self.width] # (h, w)
        # Spatial coordinates of splat, shape (h, w)
        splat_yy = (yy / self.space_sigma + .5).astype(np.int32) + self.padding_xy
        splat_xx = (xx / self.space_sigma + .5).astype(np.int32) + self.padding_xy
        # Spatial coordinates of slice, shape (h, w)
        slice_yy = yy.astype(np.float32) / self.space_sigma + self.padding_xy
        slice_xx = xx.astype(np.float32) / self.space_sigma + self.padding_xy

        # Spatial interpolation index of slice
        y_index, yy_index = get_both_indices(small_height, slice_yy) # (h, w)
        x_index, xx_index = get_both_indices(small_width, slice_xx) # (h, w)
        
        # Spatial interpolation factor of slice
        y_alpha = (slice_yy - y_index).reshape((-1, )) # (h x w, )
        x_alpha = (slice_xx - x_index).reshape((-1, )) # (h x w, )

        if not self.is_bilateral:
            # Spatial convolutional dimension
            self.dim = 2

            # Spatial interpolation index and factor
            interp_indices = np.asarray([y_index, yy_index, x_index, xx_index]) # (10, h x w)
            alphas = np.asarray([1. - y_alpha, y_alpha, 1. - x_alpha, x_alpha]) # (10, h x w)

            # Method of coordinate transformation
            def coord_transform(idx):
                return (idx[:, 0, :] * small_width + idx[:, 1, :]).reshape((-1, )) # (2^dim x h x w, )

            # Initialize interpolation
            offset = np.arange(self.dim) * 2 # [dim, ]
            # Permutation
            permutations = np.asarray(list(it.product(range(2), repeat=self.dim))).reshape((-1, self.dim))
            permutations += offset[np.newaxis, ...]
            permutations = permutations.reshape((-1, )) # Flatten, [2^dim x dim]
            alpha_prods = alphas[permutations].reshape((-1, self.dim, self.size)) # [2^dim, dim, h x w]
            idx = interp_indices[permutations].reshape((-1, self.dim, self.size)) # [2^dim, dim, h x w]

            # Shape of spatial data grid
            self.data_shape = (small_height, small_width)
            
            # Spatial splat coordinates, shape (h x w, )
            self.splat_coords = (splat_yy * small_width + splat_xx).reshape((-1, ))

            # Interpolation indices and alphas of spatial slice
            self.slice_idx = coord_transform(idx)
            self.alpha_prod = alpha_prods.prod(axis=1)

        else:
            # Bilateral convolutional dimension
            self.dim = 5

            # Decompose `features` into r, g, and b channels
            r, g, b = features[..., 0], features[..., 1], features[..., 2]
            r_min, r_max = r.min(), r.max()
            g_min, g_max = g.min(), g.max()
            b_min, b_max = b.min(), b.max()
            r_delta, g_delta, b_delta = r_max - r_min, g_max - g_min, b_max - b_min
            # Range coordinates, shape (h, w), dtype float
            rr, gg, bb = r - r_min, g - g_min, b - b_min

            # Depth of grid
            small_rdepth = self.size_type(r_delta / self.range_sigma) + 1 + 2 * self.padding_z 
            small_gdepth = self.size_type(g_delta / self.range_sigma) + 1 + 2 * self.padding_z 
            small_bdepth = self.size_type(b_delta / self.range_sigma) + 1 + 2 * self.padding_z

            # Range coordinates, shape (h, w)
            splat_rr = (rr / self.range_sigma + .5).astype(np.int32) + self.padding_z
            splat_gg = (gg / self.range_sigma + .5).astype(np.int32) + self.padding_z
            splat_bb = (bb / self.range_sigma + .5).astype(np.int32) + self.padding_z

            # Range coordinates, shape (h, w)
            slice_rr = rr / self.range_sigma + self.padding_z
            slice_gg = gg / self.range_sigma + self.padding_z
            slice_bb = bb / self.range_sigma + self.padding_z

            # Slice interpolation range coordinate pairs
            r_index, rr_index = get_both_indices(small_rdepth, slice_rr) # (h, w)
            g_index, gg_index = get_both_indices(small_gdepth, slice_gg) # (h, w)
            b_index, bb_index = get_both_indices(small_bdepth, slice_bb) # (h, w)

            # Interpolation factors
            r_alpha = (slice_rr - r_index).reshape((-1, )) # (h x w, )
            g_alpha = (slice_gg - g_index).reshape((-1, )) # (h x w, )
            b_alpha = (slice_bb - b_index).reshape((-1, )) # (h x w, )

            # Bilateral interpolation index and factor
            interp_indices = np.asarray([y_index, yy_index, x_index, xx_index, r_index, rr_index, g_index, gg_index, b_index, bb_index]) # (10, h x w)
            alphas = np.asarray([1. - y_alpha, y_alpha, 1. - x_alpha, x_alpha, 1. - r_alpha, r_alpha, 1. - g_alpha, g_alpha, 1. - b_alpha, b_alpha]) # (10, h x w)

            # Method of coordinate transformation
            def coord_transform(idx):
                return ((((idx[:, 0, :] * small_width + idx[:, 1, :]) * small_rdepth + idx[:, 2, :]) * small_gdepth + idx[:, 3, :]) * small_bdepth + idx[:, 4, :]).reshape((-1, )) # (2^dim x h x w, )

            # Initialize interpolation
            offset = np.arange(self.dim) * 2 # [dim, ]
            # Permutation
            permutations = np.asarray(list(it.product(range(2), repeat=self.dim))).reshape((-1, self.dim))
            permutations += offset[np.newaxis, ...]
            permutations = permutations.reshape((-1, )) # Flatten, [2^dim x dim]
            alpha_prods = alphas[permutations].reshape((-1, self.dim, self.size)) # [2^dim, dim, h x w]
            idx = interp_indices[permutations].reshape((-1, self.dim, self.size)) # [2^dim, dim, h x w]

            # Bilateral grid shape
            self.data_shape = (small_height, small_width, small_rdepth, small_gdepth, small_bdepth)

            # Bilateal splat coordinates, shape (h x w, )
            self.splat_coords = ((((splat_yy * small_width + splat_xx) * small_rdepth + splat_rr) * small_gdepth + splat_gg) * small_bdepth + splat_bb).reshape((-1, ))

            # Interpolation indices and alphas of bilateral slice
            self.slice_idx = coord_transform(idx)
            self.alpha_prod = alpha_prods.prod(axis=1)
        
        # Bilateral grid and buffer
        self.data = np.zeros(self.data_shape, dtype=np.float32) # For each channel
        self.data_flat = self.data.reshape((-1, )) # view of data
        self.buffer = np.zeros_like(self.data)

        # Interpolation grid
        self.interpolation = np.zeros((self.height * self.width, ), dtype=np.float32)

    def convn(self, n_iter: int=2):
        self.buffer.fill(0)
        perm = list(range(1, self.data.ndim)) + [0] # [1, ..., ndim - 1, 0] 

        for _ in range(n_iter):
            self.buffer, self.data = self.data, self.buffer

            for dim in range(self.data.ndim):
                self.data[1:-1] = (self.buffer[:-2] + self.buffer[2:] + 2. * self.buffer[1:-1]) / 4.
                self.data = np.transpose(self.data, perm)
                self.buffer = np.transpose(self.buffer, perm)

    def Nlinear_interpolation(self) -> np.ndarray:
        # Initialize interpolation
        self.interpolation.fill(0)

        self.interpolation[:] = (self.alpha_prod * self.data_flat[self.slice_idx].reshape((-1, self.size))).sum(axis=0)

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
            self.Nlinear_interpolation()

            # Get result0
            out_ch[:] = self.interpolation.reshape((self.height, self.width))
