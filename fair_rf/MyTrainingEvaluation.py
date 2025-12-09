
import Chromosome
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomTreesEmbedding, StackingClassifier, VotingClassifier, HistGradientBoostingClassifier
import copy
from sklearn.utils import resample
import pandas as pd
from sklearn.metrics import accuracy_score,recall_score,precision_score, f1_score,matthews_corrcoef
from aif360.metrics import ClassificationMetric
import warnings
import time
import datetime
from mydatautil import *
from metrics import Metrics
from mydatautil import Dataset
from sklearn.model_selection import train_test_split
import csv
from PretrainedVotingClassifier import PretrainedVotingClassifier
from PretrainedStackingClassifier import PretrainedStackingClassifier

warnings.simplefilter('ignore', UserWarning)

class MyTrainingEvaluation():

    def __init__(self, obj_1, obj_2, df_train, df_test, privileged_group, unprivileged_group, dataset, protected_attribute):
        self.obj_1 = obj_1
        self.obj_2 = obj_2
        self.df_train = copy(df_train)
        self.df_test = copy(df_test)
        self.privileged_group = privileged_group
        self.unprivileged_group = unprivileged_group
        self.dataset = dataset
        self.protected_attribute = protected_attribute
        n_obj = 2

    def mutate_df(self, data: Dataset, mutation_percentage):
        df_copy = copy.copy(data.df)
        print("Before mutation:", df_copy.equals(data.df))
        num_rows = len(df_copy)
        num_mutations = int(num_rows * mutation_percentage)
        mutation_indices = np.random.choice(num_rows, num_mutations, replace=False)
        
        # Flip values for each protected attribute in the privileged groups
        for protected_attr in data.privileged_groups.keys():
            if protected_attr in df_copy.columns:
                col_idx = data.df.columns.get_loc(protected_attr)
                # Flip the values (0 -> 1, 1 -> 0) for the selected mutation indices
                df_copy.iloc[mutation_indices, col_idx] = 1 - df_copy.iloc[mutation_indices, col_idx]
        print("After mutation:", df_copy.equals(data.df))
        return df_copy

    def run_models(self, chrom):
        print("Running models for chromosome:", chrom.__dict__)
        # ensemble_estimators_list = []
        # mutation_list = []
        # for model,mutation in zip(chrom.model_list, chrom.mutation_list):
        #     # print("THIS IS THE MODEL", model.__dict__)
        #     if model.is_on == True:
        #         ensemble_estimators_list.append((model.name, model.ml_model))
        #         mutation_list.append(mutation)
        # print("ENSEMBLE ESTIMATORS LIST ", ensemble_estimators_list)
        # for model, mutation in zip(ensemble_estimators_list, mutation_list):
        df_train = MyTrainingEvaluation.mutate_df(self, self.df_train, chrom.mutation_list)
        model = chrom.model.__dict__['ml_model']
        model.fit(df_train.drop(columns=self.df_train.label_name), df_train[self.df_train.label_name])
        # if chrom.ensemble_strategy == "Stacking":
        #     ensemble_model = PretrainedStackingClassifier(estimators=ensemble_estimators_list)
        #     ensemble_model.fit(df_train.drop(columns=self.df_train.label_name), df_train[self.df_train.label_name])
        # elif chrom.ensemble_strategy == "Hard_Voting":
        #     ensemble_model = PretrainedVotingClassifier(estimators=ensemble_estimators_list)
        # else:
        #     if chrom.ensemble_strategy == "Soft_Voting":
        #         ensemble_model = PretrainedVotingClassifier(voting='soft', estimators=ensemble_estimators_list)
        
        # print(self.df_train.labels)
        # fitted_model = ensemble_model.fit(df_train.drop(columns=self.df_train.label_name), df_train[self.df_train.label_name])
        pred = model.predict(self.df_test.df.drop(columns=self.df_train.label_name))
        pred = [[i] for i in pred]
        train_df_copy = copy.copy(self.df_test.df)
        train_df_copy['pred'] = np.array(pred)
        # if (len(ensemble_estimators_list)==0):
        #     print("******WARNING: STACKING EMPTY******")
        return self.df_test.df, train_df_copy

    def measure_train_score(self, chrom, val_df: pd.DataFrame, val_df_predictions: pd.DataFrame):

        y_test = val_df[self.df_train.label_name]
        y_pred = val_df_predictions['pred']
        # val_df['pred'] = y_pred
        accuracy = round(accuracy_score(y_test, y_pred), 3)
        recall_macro = round(recall_score(y_test, y_pred, average='macro'), 3)
        precision_macro = round(precision_score(y_test, y_pred, average='macro'), 3)
        f1score_macro = round(f1_score(y_test, y_pred, average='macro'), 3)
        mcc = round(matthews_corrcoef(y_test, y_pred), 3)

        metrics = Metrics(val_df_predictions, 'pred', self.df_train.label_name, self.df_train.pos_class)

        # classified_metric_pred = ClassificationMetric(dataset_orig_train, train_data_predictions,
        #                                             unprivileged_groups=unprivileged_group,
        #                                             privileged_groups=privileged_group)
        spd = round(abs(metrics.statistical_parity(self.df_train.unprivileged_groups)), 3)
        aod = round(abs(metrics.average_odds(self.df_train.unprivileged_groups)), 3)
        eod = round(abs(metrics.equal_opportunity(self.df_train.unprivileged_groups)), 3)
        # spd, aod, eod = 0.2

        chrom.score = {'accuracy': accuracy, 'recall': recall_macro, 'precision': precision_macro, 'f1': f1score_macro, 'mcc': mcc, 'spd': spd, 'aod': aod, 'eod': eod}
        return chrom