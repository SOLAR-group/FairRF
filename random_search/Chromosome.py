## import ML_model class 

import numpy as np
from mLModel import mLModel
# from sklearn.ensemble import *


class Chromosome():
  
    is_changed = bool
    score = {}
    ensemble_strategy = ""
    model_list = []
    # self._X = np.array()
    # def __init__(self, ensemble_strategy, knn_model, svm_model, rf_model, cart_model, lr_model):
    #     self.is_changed = True
    #     self.score = {}
    #     self.ensemble_strategy = ensemble_strategy     
    #     self.model_list = [knn_model, svm_model, rf_model, cart_model, lr_model]

    def __init__(self, ensemble_strategy, knn_model, rf_model, cart_model, lr_model):
        self.is_changed = True
        self.score = {}
        self.ensemble_strategy = ensemble_strategy     
        self.model_list = [knn_model, rf_model, cart_model, lr_model]
        # self._X= self, '_F': array([]), '_G': array([], dtype=float64), '_H': array([], dtype=float64), '_dF': array([], dtype=float64), '_dG': array([], dtype=float64), '_dH': array([], dtype=float64), '_ddF': array([], dtype=float64), '_ddG': array([], dtype=float64), '_ddH': array([], dtype=float64), '_CV': array([0.]), 'evaluated': {'H', 'F', 'G'}, 'data': {'n_gen': 1, 'n_iter': 1, 'rank': 0, 'crowding': inf}, 'config': {'cache': True, 'cv_eps': 0.0, 'cv_ieq': {'scale': None, 'eps': 0.0, 'pow': None, 'func': <function sum at 0x102cd8670>}, 'cv_eq': {'scale': None, 'eps': 0.0001, 'pow': None, 'func': <function sum at 0x102cd8670>