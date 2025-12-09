from sklearn.base import BaseEstimator

class mLModel(object):

    def __init__(self):
        self.name = ""
        self.is_on = True
        self.file_path = ""
        self.ml_model = BaseEstimator
        self.hyper_params = {}
        # self.param_ranges = {self.ml_model.get_params()}
        self.param_ranges = {}


    def __init__(self, name, is_on, file_path, ml_model, hyper_params, param_ranges):
        self.name = name
        self.is_on = True
        self.file_path = file_path
        self.ml_model = ml_model
        self.hyper_params = hyper_params
        self.param_ranges = param_ranges

    def create_model(self, name, ml_model, param_ranges):
        self.name = name
        self.is_on = True
        self.file_path = ""
        self.ml_model = ml_model
        self.hyper_params = ml_model.get_params()
        self.param_ranges = param_ranges