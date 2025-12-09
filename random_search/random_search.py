import random
import datetime
import warnings
import os
import numpy as np
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomTreesEmbedding, StackingClassifier, VotingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn import svm
from sklearn import tree
# from Chromosome import Chromosome
from mLModel import mLModel

# from MyCrossover import *
from mydatautil import *
from SolutionEvaluation import *

import copy
import time
from sklearn.metrics import accuracy_score,recall_score,precision_score, f1_score,matthews_corrcoef
from aif360.metrics import ClassificationMetric
import matplotlib.pyplot as plt
from deap.benchmarks.tools import diversity, convergence, hypervolume
import csv


class RandomSearch():    
    POP_SIZE = 6
     
    def __init__(self, dataset,protected_attribute,round):
        self.dataset = dataset
        self.protected_attribute = protected_attribute
        warnings.simplefilter('ignore', UserWarning)
        # my_seed = random.randint(1, 2**32 - 1)
        my_seed = round 
        self.df_train, self.df_test, self.privileged_group, self.unprivileged_group = read_data(
            self.dataset, self.protected_attribute, my_seed
        )
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0)) # Maximise Performance and Minimise Bias        
        creator.create("Individual", dict, is_changed = True, score = {}, ensemble_strategy = "", model_list = [], 
                       mutation_list=[], fitness=creator.FitnessMulti)       

    def createInd(self):    
        individual = creator.Individual() 
        # Create the first model_pool and inizialise the hypermparamenters
        initial_model_pool = [
                mLModel('knn', True, 'testing_path', KNeighborsClassifier(), KNeighborsClassifier().get_params(), {'n_neighbors': [2,3,4,5,6,8,10,12,14,18,20], 'weights' :['uniform', 'distance'], 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'], 'p': [1, 2]}),
                mLModel('rf',  True, "testing_path", RandomForestClassifier(), RandomForestClassifier().get_params(), {'n_estimators':[10, 20, 50, 80, 100, 150, 200], 'criterion' : ["gini", "entropy", "log_loss"], 'max_depth' : [None, 10, 15, 20, 30, 40, 50], 'min_samples_split' : [2, 3, 4], 'max_features': ['sqrt', 'log2', None]}),
                mLModel('cart', True, "testing_path" , tree.DecisionTreeClassifier(), tree.DecisionTreeClassifier().get_params(), {'criterion' : ["gini", "entropy", "log_loss"], 'max_depth' : [None, 10, 15, 20, 30, 40, 50], 'splitter': ['best', 'random'], 'max_features': ['sqrt', 'log2', None]}),
                mLModel('lr', True, "testing_path" , LogisticRegression(), LogisticRegression().get_params(), {'fit_intercept' :[True, False], 'class_weight': ["balanced", None], "solver" :["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"]})
                ]
        ensemble_strategy_models = ["Stacking", "Hard_Voting", "Soft_Voting"]
        individual.is_changed = True
        individual.score = {'accuracy': None, 'recall': None, 'precision': None, 'f1': None, 'mcc': None, 'spd': None, 'aod': None, 'eod': None}
        individual.ensemble_strategy = random.choice(ensemble_strategy_models)
        individual.model_list = [initial_model_pool[0], initial_model_pool[1],initial_model_pool[2],initial_model_pool[3]]
        # individual.mutation_num = random.uniform(0.0, 1.0)

        for model_ind, model_name in enumerate(individual.model_list):
            model = individual.model_list[model_ind]
            for param_name, param_value in model.param_ranges.items():
                model.hyper_params.update({param_name: random.choice(param_value)})
            model.ml_model.set_params(**model.hyper_params)  
       
       
        # Create additional models and append them to the chromosome model_pool
        model_num = random.randint(0, 5)
        for model_count in range(0, model_num):
            model_pool = [
                mLModel('knn', True, 'testing_path', KNeighborsClassifier(), KNeighborsClassifier().get_params(), {'n_neighbors': [2,3,4,5,6,8,10,12,14,18,20], 'weights' :['uniform', 'distance'], 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'], 'p': [1, 2]}),
                mLModel('rf',  True, "testing_path", RandomForestClassifier(), RandomForestClassifier().get_params(), {'n_estimators':[10, 20, 50, 80, 100, 150, 200], 'criterion' : ["gini", "entropy", "log_loss"], 'max_depth' : [None, 10, 15, 20, 30, 40, 50], 'min_samples_split' : [2, 3, 4], 'max_features': ['sqrt', 'log2', None]}),
                mLModel('cart', True, "testing_path" , tree.DecisionTreeClassifier(), tree.DecisionTreeClassifier().get_params(), {'criterion' : ["gini", "entropy", "log_loss"], 'max_depth' : [None, 10, 15, 20, 30, 40, 50], 'splitter': ['best', 'random'], 'max_features': ['sqrt', 'log2', None]}),
                mLModel('lr', True, "testing_path" , LogisticRegression(), LogisticRegression().get_params(), {'fit_intercept' :[True, False], 'class_weight': ["balanced", None], "solver" :["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"]})
            ]
            
            individual.model_list.append(random.choice(model_pool))
            
            individual.model_list[-1].name = individual.model_list[-1].name + "_" + str(model_count)
            model_num = len(individual.model_list)
            rand_list = [round(random.uniform(0,1)) for _ in range(model_num)]
            
            # while all(item == 0 for item in rand_list):
            #     rand_list = [random.randint(0,1) for _ in range(model_num)]
            
            while rand_list.count(1) < 2:
                # rand_list = [random.randint(0,1) for _ in range(model_num)]
                rand_list = [round(random.uniform(0,1)) for _ in range(len(individual.model_list))]
            
            # print("THIS IS THE RANDOM LIST ", rand_list)
            
            random.shuffle(rand_list)
            #inizialise the additional models on/of + hypermparamenters
            for gene in individual.keys():
                if gene == 'model_list':
                    for model_ind in range(len(individual.model_list)):
                    # for model_ind, model_name in enumerate(getattr(chrom,'model_list')):
                        if rand_list[model_ind] == 0:
                            model = individual.model_list[model_ind].__dict__
                            model.is_on = False
                            for param_name, param_value in model.param_ranges.items():
                                model.hyper_params.update({param_name: random.choice(param_value)})
                            model.ml_model.set_params(**model.hyper_params)  
                        else:
                            if rand_list[model_ind] == 1:
                                model = individual.model_list[model_ind].__dict__
                                model.is_on = True
                                #inizialise the hypermparamenters
                                for param_name, param_value in model.param_ranges.items():
                                    model.hyper_params.update({param_name: random.choice(param_value)})
                                model.ml_model.set_params(**model.hyper_params) 
        individual.mutation_list = [round(random.uniform(0.0, 1.0), 2) for _ in range(len(individual.model_list))]
        return individual
    

    # dataset: adult        # protected attributes: sex, race 
    # dataset: german        # protected attributes: sex, age 
    # dataset: compas        # protected attributes: sex, race 
    # dataset: bank        # protected attributes: age
    # dataset: mep        # protected attributes: RACE 


if __name__ == "__main__":
    
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--dataset', '-d')
    parser.add_argument('--start_round', '-r', default=0, type=int)

    args = parser.parse_args()
        
    if args.dataset == 'adult':
        design_dict = {'adult': ['sex', 'race']}
    elif args.dataset == 'german':
        design_dict = {'german': ['sex', 'age']} 
    elif args.dataset == 'compas':
        design_dict = {'compas': ['sex', 'race']}
    elif args.dataset == 'bank':
        design_dict = {'bank': ['age']}
    elif args.dataset == 'mep':
        design_dict = {'mep': ['RACE']}
    else:
        design_dict = {
            # 'adult': ['sex', 'race'],
            # 'german': ['sex', 'age'],
            # 'compas': ['sex', 'race'],
            'bank': ['age'],
            # 'cmc': ['wife_religion'],
            # 'mep': ['RACE'],
            # 'crime': ['black_people'],
            # 'drug': ['race'],
            # 'law': ['race'],
            # 'park': ['sex']
        }

    for dataset in design_dict.keys():
        for attr in design_dict[dataset]:
            for run_num in range(args.start_round, 20):
                rs = RandomSearch(dataset, attr, run_num)
                pop = [rs.createInd() for _ in range(rs.POP_SIZE)]

                testing_start = time.perf_counter()

                testing_pareto_chroms = evaluate_solutions_testing(pop, 
                                                                   rs.df_train, 
                                                                   rs.df_test, 
                                                                   rs.privileged_group, 
                                                                   rs.unprivileged_group)
                os.makedirs(f'outputs/{rs.dataset}_{rs.protected_attribute}', exist_ok=True)    
                with open(f"outputs/{rs.dataset}_{rs.protected_attribute}/"+rs.dataset + "_testset_" + rs.protected_attribute + '_run_' + str(run_num) + '.txt', 'w') as f: 
                    for pareto in testing_pareto_chroms:
                        f.write('\n***************************Chromo*************************** \n')
                        print('***CHROMO***', pareto.__dict__)
                        # for key1, value1 in pareto[0].__dict__.items():
                        for vals in pareto.__dict__:
                            if vals == 'score' or vals == 'ensemble_strategy': 
                                f.write('%s:%s\n' % (vals, pareto.__dict__[vals]))
                        for model in pareto.model_list:
                            model_dict = model.__dict__.items()
                            for key, value in model_dict:
                                if key == 'name' or key == 'ml_model' or key == 'hyper_params' or key == 'param_ranges':
                                    f.write('%s:%s\n' % (key, value))
                
        
