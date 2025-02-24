import math

import numpy as np
import scipy.ndimage as ndi

from .._shared.filters import gaussian
from .._shared.utils import _supported_float_type, check_nD
from ..feature.util import DescriptorExtractor, FeatureDetector
from ..transform import rescale

def _edgeness(hxx, hyy, hxy): ...
def _sparse_gradient(vol, positions): ...
def _hessian(d, positions): ...
def _offsets(grad, hess): ...

class SIFT(FeatureDetector, DescriptorExtractor):
    def __init__(
        self,
        upsampling: int = 2,
        n_octaves: int = 8,
        n_scales: int = 3,
        sigma_min: float = 1.6,
        sigma_in: float = 0.5,
        c_dog: float = ...,
        c_edge: float = 10,
        n_bins: int = 36,
        lambda_ori: float = 1.5,
        c_max: float = 0.8,
        lambda_descr: float = 6,
        n_hist: int = 4,
        n_ori: int = 8,
    ): ...
    @property
    def deltas(self): ...
    def _set_number_of_octaves(self, image_shape): ...
    def _create_scalespace(self, image): ...
    def _inrange(self, a, dim): ...
    def _find_localize_evaluate(self, dogspace, img_shape): ...
    def _fit(self, h): ...
    def _compute_orientation(self, positions_oct, scales_oct, sigmas_oct, octaves, gaussian_scalespace): ...
    def _rotate(self, row, col, angle): ...
    def _compute_descriptor(self, gradient_space): ...
    def _preprocess(self, image): ...
    def detect(self, image): ...
    def extract(self, image): ...
    def detect_and_extract(self, image): ...
