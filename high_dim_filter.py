'''
High-dimensional filter
spatial high-dim filter if `features` is None
bilateral high-dim filter if `features` is np.ndarray
'''

import numpy as np
import itertools as it

def clamp(min_value: float, max_value: float, x: np.ndarray) -> np.ndarray:
    return np.maximum(min_value, np.minimum(max_value, x))

class HighDimFilter:
    '''
    High-dimensional filter
    '''
    def __init__(self, is_bilateral: bool, height: int, width: int, space_sigma: float=16, range_sigma: float=.25, padding_xy: int=2, padding_z: int=2) -> None:
        '''
        Initializer

        Args:
            height: image height, int
            width: image width, int
            space_sigma: sigma_s, float
            range_sigma: sigma_r, float
            padding_xy: number of pixel for padding alongside y and x, int
            padding_z: number of pixel for padding alongside z, int

        Returns:
            None
        '''
        # Is bilateral?
        self.is_bilateral = is_bilateral
        # Datatype cast
        self.size_type = int
        # Index order: y --> height, x --> width, z --> depth
        self.height, self.width = height, width
        self.space_sigma, self.range_sigma = space_sigma, range_sigma
        self.padding_xy, self.padding_z = padding_xy, padding_z

    def init(self, features: np.ndarray=None) -> None:
        # Initialize a spatial high-dim filter if `features` is None; otherwise initialize a bilateral one and `features` should be three-channel and channel-last
        if self.is_bilateral:
            assert features.ndim == 3 and features.shape[-1] == 3

        # Height and width of grid, scala, dtype size_type
        self.small_height = self.size_type((self.height - 1) / self.space_sigma) + 1 + 2 * self.padding_xy 
        self.small_width = self.size_type((self.width - 1) / self.space_sigma) + 1 + 2 * self.padding_xy

        # Space coordinates, shape (h, w), dtype int
        yy, xx = np.mgrid[:self.height, :self.width] # (h, w)
        # Spatial coordinates of splat, shape (h, w)
        splat_yy = (yy / self.space_sigma + .5).astype(np.int32) + self.padding_xy
        splat_xx = (xx / self.space_sigma + .5).astype(np.int32) + self.padding_xy
        # Spatial coordinates of slice, shape (h, w)
        slice_yy = yy.astype(np.float32) / self.space_sigma + self.padding_xy
        slice_xx = xx.astype(np.float32) / self.space_sigma + self.padding_xy

        # Left and right indices of the interpolation
        def get_both_indices(size, coord):
            left_index = clamp(0, size - 1, coord.astype(np.int32))
            right_index = clamp(0, size - 1, left_index + 1)
            return left_index, right_index

        # Spatial interpolation index of slice
        y_index, yy_index = get_both_indices(self.small_height, slice_yy) # (h, w)
        x_index, xx_index = get_both_indices(self.small_width, slice_xx) # (h, w)
        
        # Spatial interpolation factor of slice
        y_alpha = (slice_yy - y_index).reshape((-1, )) # (h x w, )
        x_alpha = (slice_xx - x_index).reshape((-1, )) # (h x w, )

        if not self.is_bilateral:
            # Spatial grid shape
            self.data_shape = (self.small_height, self.small_width)
            
            # Spatial splat coordinates, shape (h x w, )
            self.splat_coords = splat_yy * self.small_width + splat_xx
            self.splat_coords = self.splat_coords.reshape((-1, )) # (h x w, )

            # Spatial interpolation index and factor
            self.left_indices = [y_index, x_index]
            self.right_indices = [yy_index, xx_index]
            self.alphas = [y_alpha, x_alpha] # (2, h x w)

            # Spatial convolutional dimension
            self.dim = 2
        else:
            # Decompose `features` into r, g, and b channels
            r, g, b = features[..., 0], features[..., 1], features[..., 2]
            r_min, r_max = r.min(), r.max()
            g_min, g_max = g.min(), g.max()
            b_min, b_max = b.min(), b.max()
            r_delta, g_delta, b_delta = r_max - r_min, g_max - g_min, b_max - b_min
            # Range coordinates, shape (h, w), dtype float
            rr, gg, bb = r - r_min, g - g_min, b - b_min

            # Depth of grid
            self.small_rdepth = self.size_type(r_delta / self.range_sigma) + 1 + 2 * self.padding_z 
            self.small_gdepth = self.size_type(g_delta / self.range_sigma) + 1 + 2 * self.padding_z 
            self.small_bdepth = self.size_type(b_delta / self.range_sigma) + 1 + 2 * self.padding_z

            # Range coordinates, shape (h, w)
            splat_rr = (rr / self.range_sigma + .5).astype(np.int32) + self.padding_z
            splat_gg = (gg / self.range_sigma + .5).astype(np.int32) + self.padding_z
            splat_bb = (bb / self.range_sigma + .5).astype(np.int32) + self.padding_z

            # Range coordinates, shape (h, w)
            slice_rr = rr / self.range_sigma + self.padding_z
            slice_gg = gg / self.range_sigma + self.padding_z
            slice_bb = bb / self.range_sigma + self.padding_z

            # Slice interpolation range coordinate pairs
            r_index, rr_index = get_both_indices(self.small_rdepth, slice_rr) # (h, w)
            g_index, gg_index = get_both_indices(self.small_gdepth, slice_gg) # (h, w)
            b_index, bb_index = get_both_indices(self.small_bdepth, slice_bb) # (h, w)

            # Interpolation factors
            r_alpha = (slice_rr - r_index).reshape((-1, )) # (h x w, )
            g_alpha = (slice_gg - g_index).reshape((-1, )) # (h x w, )
            b_alpha = (slice_bb - b_index).reshape((-1, )) # (h x w, )

            # Bilateral grid shape
            self.data_shape = (self.small_height, self.small_width, self.small_rdepth, self.small_gdepth, self.small_bdepth)

            # Bilateal splat coordinates, shape (h x w, )
            self.splat_coords = (((splat_yy * self.small_width + splat_xx) * self.small_rdepth + splat_rr) * self.small_gdepth + splat_gg) * self.small_bdepth + splat_bb
            self.splat_coords = self.splat_coords.reshape((-1)) # (h x w, )

            # Bilateral interpolation index and factor
            self.left_indices = [y_index, x_index, r_index, g_index, b_index]
            self.right_indices = [yy_index, xx_index, rr_index, gg_index, bb_index]
            self.alphas = [y_alpha, x_alpha, r_alpha, g_alpha, b_alpha] # (5, h x w)

            # Bilateral convolutional dimension
            self.dim = 5
        
        # Bilateral grid and buffer
        self.data = np.zeros(self.data_shape, dtype=np.float32) # For each channel
        self.data_flat = self.data.reshape((-1, )) # view of data
        self.buffer = np.zeros_like(self.data)

        # Interpolation grid
        self.interpolation = np.zeros((self.height * self.width, ), dtype=np.float32)
        self.alpha_prod = np.ones((self.height * self.width, ), dtype=np.float32)

    def convn(self, n_iter: int=2):
        self.buffer.fill(0)
        perm = list(range(1, self.data.ndim)) + [0] # [1, ..., ndim - 1, 0] 

        for _ in range(n_iter):
            self.buffer, self.data = self.data, self.buffer

            for dim in range(self.data.ndim):
                self.data[1:-1] = (self.buffer[:-2] + self.buffer[2:] + 2. * self.buffer[1:-1]) / 4.
                self.data = np.transpose(self.data, perm)
                self.buffer = np.transpose(self.buffer, perm)

    def loop_Nlinear_interpolation(self) -> np.ndarray:
        # Coordinate transformation
        def set_coord_transform():
            def bilateral_coord_transform(y_idx, x_idx, r_idx, g_idx, b_idx):
                return ((((y_idx * self.small_width + x_idx) * self.small_rdepth + r_idx) * self.small_gdepth + g_idx) * self.small_bdepth + b_idx).reshape((-1, ))

            def spatial_coord_transform(y_idx, x_idx):
                return (y_idx * self.small_width + x_idx).reshape((-1))

            if self.is_bilateral:
                return bilateral_coord_transform
            else:
                return spatial_coord_transform

        coord_transform = set_coord_transform()
        
        # Initialize interpolation
        self.interpolation.fill(0)

        for perm in it.product(range(2), repeat=self.dim):
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
