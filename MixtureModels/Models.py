"""Mixture model for collaborative filtering"""
from typing import NamedTuple, Tuple
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Arc


class GaussianMixture(NamedTuple):
  """Tuple holding a gaussian mixture"""
  mu: np.ndarray  # (K, d) array - each row corresponds to a gaussian component mean
  var: np.ndarray  # (K, ) array - each row corresponds to the variance of a component
  p: np.ndarray  # (K, ) array = each row corresponds to the weight of a component

  f
