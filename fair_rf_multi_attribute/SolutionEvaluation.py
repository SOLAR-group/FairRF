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
from mydatautil import Dataset
from metrics import Metrics
from PretrainedVotingClassifier import PretrainedVotingClassifier
from PretrainedStackingClassifier import PretrainedStackingClassifier

def flatten(lis:list):
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

def mutate_df(data: Dataset, mutation_percentage):
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
    
def measure_train_score_testing(chrom, dataset_orig, data_orig_with_predictions, privileged_group, unprivileged_group):
    y_test = dataset_orig.labels
    y_pred = data_orig_with_predictions.labels
    df = dataset_orig.df
    df['pred'] = y_pred
    metrics = Metrics(df, 'pred', dataset_orig.label_name, dataset_orig.pos_class)
    accuracy = round(accuracy_score(y_test, y_pred),3)
    recall_macro = round(recall_score(y_test, y_pred, average='macro'),3)
    precision_macro = round(precision_score(y_test, y_pred, average='macro'),3)
    f1score_macro = round(f1_score(y_test, y_pred, average='macro'),3)
    mcc = round(matthews_corrcoef(y_test, y_pred),3)
    # classified_metric_pred = ClassificationMetric(dataset_orig, data_orig_with_predictions,
    #                                             unprivileged_groups=unprivileged_group,
    #                                             privileged_groups=privileged_group)
    spd_list = [round(abs(metrics.statistical_parity({k: v})), 3) for k,v in dataset_orig.unprivileged_groups.items()]
    aod_list = [round(abs(metrics.average_odds({k: v})), 3) for k,v in dataset_orig.unprivileged_groups.items()]
    eod_list = [round(abs(metrics.equal_opportunity({k: v})), 3) for k,v in dataset_orig.unprivileged_groups.items()]
    spd = round(abs(metrics.wc_spd(list(dataset_orig.unprivileged_groups.keys()))[-1]), 3)
    eod = round(abs(metrics.wc_eod(list(dataset_orig.unprivileged_groups.keys()))[-1]), 3)
    aod = round(abs(metrics.wc_aod(list(dataset_orig.unprivileged_groups.keys()))[-1]), 3)
    # spd, aod, eod = 0.2
    print('PRINTING CHROM', chrom)
    chrom.score = {'accuracy': accuracy, 'recall': recall_macro, 'precision': precision_macro, 'f1': f1score_macro, 'mcc': mcc, 'avg_spd': np.mean(spd_list), 'avg_aod': np.mean(aod_list), 'avg_eod': np.mean(eod_list), 'spd': spd, 'aod': aod, 'eod': eod}
    return chrom

def run_models_testing(chrom, train_set, test_set):
    # ensemble_estimators_list = []
    # mutation_list = []
    # for model,mutation in zip(chrom.__dict__['model_list'], chrom.__dict__['mutation_list']):
    # if chrom.model.__dict__['is_on'] == True:
    #         ensemble_estimators_list.append((chrom.model.__dict__['name'], chrom.model.__dict__['ml_model']))
    #         mutation_list.append(chrom.mutation_list)

    # for model, mutation in zip(ensemble_estimators_list, mutation_list):
    #     model[1].fit(df_train.drop(columns=train_set.label_name), df_train[train_set.label_name])
    df_train = mutate_df(train_set, chrom.mutation_list)
    model = chrom.model.__dict__['ml_model']
    model.fit(df_train.drop(columns=train_set.label_name), df_train[train_set.label_name])

        # if chrom.__dict__['ensemble_strategy'] == "Stacking":
        #     ensemble_model = PretrainedStackingClassifier(estimators=ensemble_estimators_list)
        #     ensemble_model.fit(df_train.drop(columns=train_set.label_name), df_train[train_set.label_name])
        # elif chrom.__dict__['ensemble_strategy'] == "Hard_Voting":
        #     ensemble_model = PretrainedVotingClassifier(estimators=ensemble_estimators_list)
        # else:
        #     if chrom.__dict__['ensemble_strategy'] == "Soft_Voting":
        #         ensemble_model = PretrainedVotingClassifier(voting='soft', estimators=ensemble_estimators_list)    
    # fitted_model = ensemble_model.fit(train_set.features, train_set.labels)
    pred = model.predict(test_set.features)
    pred = [[i] for i in pred]
    test_df_copy = copy.copy(test_set)
    test_df_copy.labels = np.array(pred)
    return test_df_copy

def evaluate_solutions_testing(pareto_training_list, train_set, test_set, privileged_group, unprivileged_group):
    start = time.perf_counter()
    pareto_training_list = flatten(pareto_training_list)
    pareto_training_list_lenght = len(pareto_training_list)
    store_results = []
    for i in range(pareto_training_list_lenght):
            a_chromosome = pareto_training_list[i]
            #given a chromosome run its ensemble on the given trainining set (apply bootstrapping to the training set)
            test_df_copy_with_prediction = run_models_testing(a_chromosome, train_set, test_set)
            a_chromosome = measure_train_score_testing(a_chromosome, test_set, test_df_copy_with_prediction, privileged_group, unprivileged_group)
            store_results.append(a_chromosome)
    print("Total time to evaluate fitness" + str(datetime.timedelta(seconds=round(time.perf_counter() - start))))
    return(store_results)