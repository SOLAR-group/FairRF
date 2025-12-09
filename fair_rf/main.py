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
from MyMutation import *
# from MyCrossover import *
from mydatautil import *
from SolutionEvaluation import *
from MyTrainingEvaluation import *
import copy
import time
from sklearn.metrics import accuracy_score,recall_score,precision_score, f1_score,matthews_corrcoef
from aif360.metrics import ClassificationMetric
import matplotlib.pyplot as plt
from deap.benchmarks.tools import diversity, convergence, hypervolume
import csv


class NSGAIIoptimizer():    
    POP_SIZE = 50
    # TOURNMTSIZE = 16
    obj_1 = 'accuracy'
    obj_2 = 'spd'
    # obj_3 = 'eod'
    # obj_4 = 'aod'
     
    def __init__(self, dataset,protected_attribute,round,model_name,args):
        self.dataset = dataset
        self.protected_attribute = protected_attribute
        warnings.simplefilter('ignore', UserWarning)
        # my_seed = random.randint(1, 2**32 - 1)
        my_seed = round
        self.acc_weight = args.acc_weight
        self.fair_weight = args.fair_weight
        self.df_train, self.df_test, self.privileged_group, self.unprivileged_group = read_data(
            self.dataset, self.protected_attribute, my_seed
        )
        self.model = model_name
        # problem = MyTrainingEvaluation(self.obj_1, self.obj_2, self.df_train, self.df_test, self.privileged_group, self.unprivileged_group, self.dataset, self.protected_attribute)

    def createInd(self):    
        individual = creator.Individual() 
        # Create the first model_pool and inizialise the hypermparamenters
        # initial_model_pool = [
        if self.model == 'knn':
            individual.model = mLModel('knn', True, 'testing_path', KNeighborsClassifier(), KNeighborsClassifier().get_params(), {'n_neighbors': [2,3,4,5,6,8,10,12,14,18,20], 'weights' :['uniform', 'distance'], 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'], 'p': [1, 2]})
        elif self.model == 'rf':
            individual.model = mLModel('rf',  True, "testing_path", RandomForestClassifier(), RandomForestClassifier().get_params(), {'n_estimators':[10, 20, 50, 80, 100, 150, 200], 'criterion' : ["gini", "entropy", "log_loss"], 'max_depth' : [None, 10, 15, 20, 30, 40, 50], 'min_samples_split' : [2, 3, 4], 'max_features': ['sqrt', 'log2', None]})
        elif self.model == 'cart':
            individual.model = mLModel('cart', True, "testing_path" , tree.DecisionTreeClassifier(), tree.DecisionTreeClassifier().get_params(), {'criterion' : ["gini", "entropy", "log_loss"], 'max_depth' : [None, 10, 15, 20, 30, 40, 50], 'splitter': ['best', 'random'], 'max_features': ['sqrt', 'log2', None]})
        elif self.model == 'lr':
            individual.model = mLModel('lr', True, "testing_path" , LogisticRegression(), LogisticRegression().get_params(), {'fit_intercept' :[True, False], 'class_weight': ["balanced", None], "solver" :["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"]})
        elif self.model == 'svm':
            individual.model = mLModel('svm', True, "testing_path" , svm.SVC(), svm.SVC().get_params(), {'C':[0.1, 1.0, 10.0, 100.0], 'kernel':['linear', 'poly', 'rbf', 'sigmoid'], 'degree':[2,3,4], 'gamma':['scale', 'auto']})
                # ]
        # ensemble_strategy_models = ["Stacking", "Hard_Voting", "Soft_Voting"]
        individual.is_changed = True
        individual.score = {'accuracy': None, 'recall': None, 'precision': None, 'f1': None, 'mcc': None, 'spd': None, 'aod': None, 'eod': None}
        # individual.ensemble_strategy = random.choice(ensemble_strategy_models)
        # individual.model_list = [initial_model_pool[0], initial_model_pool[1],initial_model_pool[2],initial_model_pool[3]]
        # individual.mutation_num = random.uniform(0.0, 1.0)

        # for model_ind, model_name in enumerate(individual.model_list):
        #     model = individual.model_list[model_ind]
        for param_name, param_value in individual.model.param_ranges.items():
            individual.model.hyper_params.update({param_name: random.choice(param_value)})
        individual.model.ml_model.set_params(**individual.model.hyper_params)


        # Create additional models and append them to the chromosome model_pool
        # model_num = random.randint(0, 5)
        # for model_count in range(0, model_num):
        #     model_pool = [
        #         mLModel('knn', True, 'testing_path', KNeighborsClassifier(), KNeighborsClassifier().get_params(), {'n_neighbors': [2,3,4,5,6,8,10,12,14,18,20], 'weights' :['uniform', 'distance'], 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'], 'p': [1, 2]}),
        #         mLModel('rf',  True, "testing_path", RandomForestClassifier(), RandomForestClassifier().get_params(), {'n_estimators':[10, 20, 50, 80, 100, 150, 200], 'criterion' : ["gini", "entropy", "log_loss"], 'max_depth' : [None, 10, 15, 20, 30, 40, 50], 'min_samples_split' : [2, 3, 4], 'max_features': ['sqrt', 'log2', None]}),
        #         mLModel('cart', True, "testing_path" , tree.DecisionTreeClassifier(), tree.DecisionTreeClassifier().get_params(), {'criterion' : ["gini", "entropy", "log_loss"], 'max_depth' : [None, 10, 15, 20, 30, 40, 50], 'splitter': ['best', 'random'], 'max_features': ['sqrt', 'log2', None]}),
        #         mLModel('lr', True, "testing_path" , LogisticRegression(), LogisticRegression().get_params(), {'fit_intercept' :[True, False], 'class_weight': ["balanced", None], "solver" :["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"]})
        #     ]
            
        #     individual.model_list.append(random.choice(model_pool))
            
        #     individual.model_list[-1].name = individual.model_list[-1].name + "_" + str(model_count)
        #     model_num = len(individual.model_list)
        #     rand_list = [round(random.uniform(0,1)) for _ in range(model_num)]
            
        #     # while all(item == 0 for item in rand_list):
        #     #     rand_list = [random.randint(0,1) for _ in range(model_num)]
            
        #     while rand_list.count(1) < 2:
        #         # rand_list = [random.randint(0,1) for _ in range(model_num)]
        #         rand_list = [round(random.uniform(0,1)) for _ in range(len(individual.model_list))]
            
        #     # print("THIS IS THE RANDOM LIST ", rand_list)
            
        #     random.shuffle(rand_list)
        #     #inizialise the additional models on/of + hypermparamenters
        #     for gene in individual.keys():
        #         if gene == 'model_list':
        #             for model_ind in range(len(individual.model_list)):
        #             # for model_ind, model_name in enumerate(getattr(chrom,'model_list')):
        #                 if rand_list[model_ind] == 0:
        #                     model = individual.model_list[model_ind].__dict__
        #                     model.is_on = False
        #                     for param_name, param_value in model.param_ranges.items():
        #                         model.hyper_params.update({param_name: random.choice(param_value)})
        #                     model.ml_model.set_params(**model.hyper_params)  
        #                 else:
        #                     if rand_list[model_ind] == 1:
        #                         model = individual.model_list[model_ind].__dict__
        #                         model.is_on = True
        #                         #inizialise the hypermparamenters
        #                         for param_name, param_value in model.param_ranges.items():
        #                             model.hyper_params.update({param_name: random.choice(param_value)})
        #                         model.ml_model.set_params(**model.hyper_params) 
        individual.mutation_list = round(random.uniform(0.0, 1.0), 2)
        return individual
    
    def flatten(self, lis:list):
        flatList = []
        # Iterate with outer list
        for element in lis:
            if type(element) is list:
                # Check if type is list than iterate through the sublist
                for item in element:
                    flatList.append(item)
            else:
                flatList.append(element)
        return flatList 
     
    def my_crossover(self, chrom1, chrom2):
        list_cross_1 = [chrom1.model, chrom1.mutation_list]
        list_cross_2 = [chrom2.model, chrom2.mutation_list]
        crossover_list_1 = self.flatten(list_cross_1)
        crossover_list_2 = self.flatten(list_cross_2)
        # print("list_cross_1: ", list_cross_1)
        # print("list_cross_2: ", list_cross_2)
        
        cutpoint = random.randrange(1, min(len(crossover_list_1), len(crossover_list_2)))
        
        offspring_1 = [crossover_list_1[0:cutpoint], crossover_list_2[cutpoint: len(crossover_list_2)]]
        offspring_1 = self.flatten(offspring_1)
        offspring_2 = [crossover_list_2[0:cutpoint], crossover_list_1[cutpoint: len(crossover_list_1)]]
        offspring_2 = self.flatten(offspring_2)
        # print("OFFSPRING 1: ", offspring_1)
        # print("OFFSPRING 2: ", offspring_2)
        chrom_offspring_1, chrom_offspring_2 = copy.copy(chrom1), copy.copy(chrom2)
        
        chrom_offspring_1.is_changed = True 
        chrom_offspring_1.model = offspring_1[0]
        chrom_offspring_1.model.is_on = True
        chrom_offspring_1.mutation_list =  offspring_1[1]
        

        chrom_offspring_2.is_changed = True 
        chrom_offspring_2.model = offspring_2[0]
        chrom_offspring_2.model.is_on = True
        chrom_offspring_2.mutation_list =  offspring_2[1]

        # print("PRINTING offspring_1", offspring_1)
        # print("PRINTING offspring_2", offspring_1)
        # chrom1_models = chrom_offspring_1.model
        # print("CHROM 1 MODELS: ", chrom1_models)
        # while all(item.is_on == False for item in chrom1_models):
        #     random.choice(chrom1_models).is_on = True
        # chrom_offspring_1.model_list = chrom1_models
        
        # chrom2_models = chrom_offspring_2.model_list
        # while all(item.is_on == False for item in chrom2_models):
        #     random.choice(chrom2_models).is_on = True
        # chrom_offspring_2.model_list = chrom2_models
    
        return chrom_offspring_1, chrom_offspring_2   
     
    
    def evaluateFitness(self, chrom):  	
        start = time.perf_counter()
        # if chrom.is_changed == True:
            #given a chromosome run its ensemble on the given trainining set (apply bootstrapping to the training set)
            # print("RUNNING models")
        val_df, val_df_copy_with_prediction = MyTrainingEvaluation.run_models(self, chrom)
        # print("EVALUATING MEASURES")
        chrom = MyTrainingEvaluation.measure_train_score(
            self, chrom, val_df, val_df_copy_with_prediction)
        # if (chrom.score[self.obj_1] > 0):
        #     f1 = round(chrom.score[self.obj_1],3)
        # else: 
        #     f1= 1000
        print("Chromosome fitness values: ", chrom.score)
        f1 = round(chrom.score[self.obj_1],3)
        
        best_value = get_fairness_thresholds(self.dataset, self.protected_attribute, self.obj_2)
        if chrom.score[self.obj_2] > best_value: 
            f2 = round(chrom.score[self.obj_2] * 2, 3)
        else:
            f2 = round(chrom.score[self.obj_2],3)
        chrom.is_changed = False
        os.makedirs(f'outputs_{args.model}/{self.dataset}_{self.protected_attribute}/fitnesses', exist_ok=True)
        with open(f'outputs_{args.model}/{self.dataset}_{self.protected_attribute}/fitnesses/' + self.dataset + '_' + self.protected_attribute + '_fitnesses.csv', 'a') as f:
            writer = csv.DictWriter(f, fieldnames=chrom.score.keys())
            if f.tell() == 0: 
                writer.writeheader()
            writer.writerow(chrom.score)
        print("Total time to evaluate fitness" + str(datetime.timedelta(seconds=round(time.perf_counter() - start))))

        return f1, f2
    
    # def evaluate_solutions(self, pareto_training_list, test_set, privileged_group, unprivileged_group):
    #     start = time.perf_counter()
    #     pareto_training_list = self.flatten(pareto_training_list)
    #     pareto_training_list_lenght = len(pareto_training_list)
    #     store_results = []
    #     for i in range(pareto_training_list_lenght):
    #             a_chromosome = pareto_training_list[i]
    #             #given a chromosome run its ensemble on the given trainining set (apply bootstrapping to the training set)
    #             test_df_copy_with_prediction = self.run_models(a_chromosome, test_set)
    #             a_chromosome = self.measure_train_score(a_chromosome, test_set, test_df_copy_with_prediction, privileged_group, unprivileged_group)
    #             store_results.append(a_chromosome)
    #     print("Total time to evaluate fitness" + str(datetime.timedelta(seconds=round(time.perf_counter() - start))))
    #     return(store_results)
    
  
    def dominates(self,x,y):
        return tools.emo.isDominated(x.fitness.values, y.fitness.values)   
    
    def main(self):
        # self.find_tools()

        NGEN = 25
        CXPB = 0.6
        MU = 5
        LAMBDA = 5

        creator.create("FitnessMulti", base.Fitness, weights=(self.acc_weight, -self.fair_weight)) # Maximise Performance and Minimise Bias
        # creator.create("Individual", dict, is_changed = True, score = {}, ensemble_strategy = "", model_list = [], fitness=creator.FitnessMulti)

        creator.create("Individual", dict, is_changed = True, score = {}, model = None, mutation_list=0, fitness=creator.FitnessMulti)

        toolbox = base.Toolbox()
        toolbox.register("Individual", self.createInd)
        toolbox.register("population", tools.initRepeat, list, toolbox.Individual)
        

        toolbox.register("evaluate", self.evaluateFitness)
        toolbox.register("mate", self.my_crossover)
        toolbox.register("mutate", MyMutation.my_mutation)
        toolbox.register("select", tools.selNSGA2)
        
        stats=tools.Statistics(key=lambda ind: ind.fitness.values)
        stats.register("avg",np.mean,axis=0)
        stats.register("std",np.std,axis=0)
        stats.register("min",np.min,axis=0)
        stats.register("max",np.max,axis=0)
        
        # logbook = tools.Logbook()
        # logbook.header = "gen", "evals", "std", "min", "avg", "max"
   
        pop = toolbox.population(n=self.POP_SIZE)
        print('Population created. Number of Individuals =', len(pop))
        hof = tools.ParetoFront()

        out, logbook = algorithms.eaMuCommaLambda(
            pop, 
            toolbox,
            mu=MU, 
            lambda_=LAMBDA, 
            cxpb=CXPB, 
            mutpb=0.2, 
            ngen=NGEN, 
            stats=stats, 
            halloffame=hof, 
            verbose=True
        )

        # valid_ind = [ind for ind in pop]
        # fitnesses = toolbox.map(toolbox.evaluate, valid_ind)
        # for ind, fit in zip(valid_ind, fitnesses):
        #     ind.fitness.values = fit
            

        
        # pop = toolbox.select(pop,len(pop))
                             
        
        # record = stats.compile(pop)
        # logbook.record(gen=0, evals=len(valid_ind), **record)
        # print(logbook.stream)
        
        
        
## ------------ -------------------- -------------------------------- --------------------------------
# 				      Begin the generational process
## -------------------------------- -------------------------------- --------------------------------

        # for gen in range(1, NGEN):
        #     # Vary the population
        #     pop = toolbox.select(pop,len(pop))

        #     offspring = tools.selTournamentDCD(pop, len(pop))
        #     offspring = list(map(toolbox.clone, offspring))

        #     for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
        #     #ind1, ind2 = self.my_selection(pop, self.met_complexity ,self.met_similarity, TOURNMTSIZE)
        #         if random.random() <= CXPB:
        #             toolbox.mate(ind1, ind2)
        #             del ind1.fitness.values
        #             del ind2.fitness.values
	      
        #         ind1 = toolbox.mutate(ind1)
        #         ind2 = toolbox.mutate(ind2)
        #         del ind1.fitness.values, ind2.fitness.values

        #     # Evaluate the individuals with an invalid fitness
        #     invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        #     fitnesses = map(toolbox.evaluate, invalid_ind)
        #     for ind, fit in zip(invalid_ind, fitnesses):
        #         ind.fitness.values = fit

        #     # Select the next generation population
        #     pop[:] = offspring
        #     record = stats.compile(pop)
        #     logbook.record(gen=gen, evals=len(valid_ind), **record)
        #     hof.update(pop)
        #     print(logbook.stream)
            
        #     best_ind = tools.selBest(pop,1)[0]

    

        return out, logbook, hof


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
    parser.add_argument('--model', '-m', default='rf', type=str, help='Model to use for the optimization. Default is rf (Random Forest).')
    parser.add_argument('--acc_weight', default=1.0, type=float, help='Weight for the accuracy objective. Default is 1.0.')
    parser.add_argument('--fair_weight', default=1.0, type=float, help='Weight for the fairness objective. Default is 1.0.')

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
            'adult': ['sex', 'race'],
            'german': ['sex', 'age'],
            'compas': ['sex', 'race'],
            'bank': ['age'],
            'mep': ['RACE'],
        }

    for dataset in design_dict.keys():
        for attr in design_dict[dataset]:
            for run_num in range(20):
                nsga = NSGAIIoptimizer(dataset, attr, run_num, args.model, args)
                pop, stats, non_dom = nsga.main()
                os.makedirs(f'outputs_{args.model}_{nsga.acc_weight}_{nsga.fair_weight}/{nsga.dataset}_{nsga.protected_attribute}', exist_ok=True)    
                os.makedirs(f'outputs_{args.model}_{nsga.acc_weight}_{nsga.fair_weight}/{nsga.dataset}_{nsga.protected_attribute}/logs', exist_ok=True) 
                pd.DataFrame(stats).to_csv(f'outputs_{args.model}_{nsga.acc_weight}_{nsga.fair_weight}/{nsga.dataset}_{nsga.protected_attribute}/logs/' + nsga.dataset + '_' + nsga.protected_attribute + '_stats_run_' + str(run_num) + '.csv', index=False)

                # non_dom = tools.sortNondominated(pop, k=len(pop),first_front_only=True)[0]
                plt.figure(figsize=(5,5))
                for ind in pop:
                    plt.plot(ind.fitness.values[0], 1-ind.fitness.values[1], 'k.', ms=5, alpha=0.75)
                for ind in non_dom:
                    plt.plot(ind.fitness.values[0], 1-ind.fitness.values[1], 'c.', alpha=0.75, ms=5)

                output_png_filename = nsga.dataset + '_' + nsga.protected_attribute + "_" + nsga.obj_1 + '+' + nsga.obj_2 + '_run_' + str(run_num)

                plt.savefig(f"imgs/{output_png_filename}.png")

                testing_start = time.perf_counter()

                testing_pareto_chroms = evaluate_solutions_testing(non_dom, 
                                                                   nsga.df_train, 
                                                                   nsga.df_test, 
                                                                   nsga.privileged_group, 
                                                                   nsga.unprivileged_group)

                with open(f"outputs_{args.model}_{nsga.acc_weight}_{nsga.fair_weight}/{nsga.dataset}_{nsga.protected_attribute}/" + nsga.dataset + "_testset_" + nsga.protected_attribute + "_" + nsga.obj_1 + '+' + nsga.obj_2 + '_run_' + str(run_num) + '.txt', 'w') as f:
                    for pareto in testing_pareto_chroms:
                        f.write('\n***************************Chromo*************************** \n')
                        print('***CHROMO***', pareto.__dict__)
                        # for key1, value1 in pareto[0].__dict__.items():
                        for vals in pareto.__dict__:
                            if vals == 'score' or vals == 'ensemble_strategy': 
                                f.write('%s:%s\n' % (vals, pareto.__dict__[vals]))
                        # for model in pareto.model:
                        model_dict = pareto.model.__dict__.items()
                        for key, value in model_dict:
                            if key == 'name' or key == 'ml_model' or key == 'hyper_params' or key == 'param_ranges':
                                f.write('%s:%s\n' % (key, value))
                
        
