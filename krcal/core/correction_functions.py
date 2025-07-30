import numpy as np

from invisible_cities.reco.corrections import ASectorMap
from invisible_cities.reco.corrections import maps_coefficient_getter
from invisible_cities.reco.corrections import correct_geometry_
from invisible_cities.types.symbols    import NormStrategy
from invisible_cities.reco.corrections import get_normalization_factor


def e0_xy_correction(map            : ASectorMap,
                     norm_strategy  : NormStrategy,
                     **norm_options : dict):
    """
    Temporal function to perfrom IC geometric corrections only
    """
    normalization   = get_normalization_factor(map        , norm_strategy, **norm_options)
    get_xy_corr_fun = maps_coefficient_getter (map.mapinfo, map.e0)
    def geo_correction_factor(x : np.array,
                              y : np.array) -> np.array:
        return correct_geometry_(get_xy_corr_fun(x,y))* normalization
    return geo_correction_factor
