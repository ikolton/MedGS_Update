from arguments import OptimizationParams
from scene.gaussian_model import GaussianModel
from models.flat_splatting.scene.points_gaussian_model import PointsGaussianModel


optimizationParamTypeCallbacks = {
    "gs": OptimizationParams,
}

gaussianModel = {
    "gs": GaussianModel,
}

gaussianModelRender = {
    "gs": GaussianModel,
    "pgs": PointsGaussianModel
}
