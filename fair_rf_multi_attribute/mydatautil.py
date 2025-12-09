
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from pprint import pprint
from sklearn.preprocessing import MinMaxScaler
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset

class Dataset:
    def __init__(self, df, label_name, pos_class, privileged_groups, unprivileged_groups):
        self.df = df
        self.label_name = label_name
        self.pos_class = pos_class
        self.privileged_groups = privileged_groups
        self.unprivileged_groups = unprivileged_groups
        self.features = df.drop(columns=[label_name]).values
        self.labels = df[label_name].values
        # print(label_name)
        # print(self.labels)

def get_values(dataset: str):

    '''
    Input: dataset name
    Returns: the label name, positive prediction, privileged group, unprivileged group
    '''
    if "cmc" in dataset:
        return ("contr_use", 2, ['wife_religion', 'wife_work'])
    if "crime" in dataset:
        return ("ViolentCrimesClass", 100, ['black_people', 'hispanic_people'])
    # if "drug" in dataset:
    #     return ("y", 0, {'race': 1}, {'race': 0})
    # if "law" in dataset:
    #     return ("gpa", 2, {'race': 1}, {'race': 0})
    # if "park" in dataset:
    #     return ("score_cut", 0, {'sex': 1}, {'sex': 0})



def get_data(dataset_used, preprocessed = False):
    pos_class = 1
    label_name = "Probability"
    if dataset_used == "adult":
        privileged_groups = {'sex': 1, 'race': 1}
        unprivileged_groups = {'sex': 0, 'race': 0}
        dataset_orig = AdultDataset().convert_to_dataframe()[0]
        dataset_orig.columns = dataset_orig.columns.str.replace("income-per-year", "Probability")
    elif dataset_used == "german":
        privileged_groups = {'sex': 1, 'age': 1}
        unprivileged_groups = {'sex': 0, 'age': 0}
        dataset_orig = GermanDataset().convert_to_dataframe()[0]
        dataset_orig['credit'] = np.where(dataset_orig['credit'] == 1, 1, 0)
        dataset_orig.columns = dataset_orig.columns.str.replace("credit", "Probability")
    elif dataset_used == "compas":
        privileged_groups = {'sex': 1, 'race': 1}
        unprivileged_groups = {'sex': 0, 'race': 0}
        dataset_orig = CompasDataset().convert_to_dataframe()[0]
        dataset_orig.columns = dataset_orig.columns.str.replace("two_year_recid", "Probability")
        dataset_orig['Probability'] = np.where(dataset_orig['Probability'] == 1, 0, 1)
        #dataset_orig['Probability'] = 1 - dataset_orig['Probability']  # make favorable_class as 1
    # elif dataset_used == "mep":
    #     privileged_groups = {'RACE': 1}
    #     unprivileged_groups = {'RACE': 0}
    #     dataset_orig = MEPSDataset19().convert_to_dataframe()[0]
    #     dataset_orig.rename(columns={'UTILIZATION': 'Probability'}, inplace=True)
    # elif dataset_used == "crime":
    #     dataset_orig = pd.read_csv('data_multiclass/crime_proc.csv', index_col=0)
    #     label_name, pos_class, privileged_groups, unprivileged_groups = get_values('crime')
    # elif dataset_used == 'drug':
    #     dataset_orig = pd.read_csv('data_multiclass/drug_proc.csv', index_col=0)
    #     label_name, pos_class, privileged_groups, unprivileged_groups = get_values('drug')
    # elif dataset_used == 'law':
    #     dataset_orig = pd.read_csv('data_multiclass/law_proc.csv', index_col=0)
    #     label_name, pos_class, privileged_groups, unprivileged_groups = get_values('law')
    # elif dataset_used == 'cmc':
    #     dataset_orig = pd.read_csv('data_multiclass/cmc_proc.csv', index_col=0)
    #     label_name, pos_class, privileged_groups, unprivileged_groups = get_values('cmc')
    # elif dataset_used == 'park':
    #     dataset_orig = pd.read_csv('data_multiclass/park_proc.csv', index_col=0)
    #     label_name, pos_class, privileged_groups, unprivileged_groups = get_values('park')
    return dataset_orig, privileged_groups, unprivileged_groups, pos_class, label_name



# def get_data_multi_attr(dataset_used): 
#     if dataset_used == "adult" or dataset_used == "compas":
#         protected_attr_names = ['sex', 'race']
#         privileged_groups = [[{'sex': 1}], [{'race': 1}]]
#         unprivileged_groups = [[{'sex': 0}], [{'race': 0}]]
        
#         if dataset_used == "adult":
#             dataset_orig = AdultDataset().convert_to_dataframe()[0]
#             dataset_orig.columns = dataset_orig.columns.str.replace("income-per-year", "Probability")
        
#         else:
#             dataset_orig = CompasDataset().convert_to_dataframe()[0]
#             dataset_orig.columns = dataset_orig.columns.str.replace("two_year_recid", "Probability")
#             dataset_orig['Probability'] = np.where(dataset_orig['Probability'] == 1, 0, 1)
    
#     return dataset_orig, protected_attr_names, privileged_groups, unprivileged_groups


    ### Uncomment the code below to run dataset from terminal ###
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-d", "--dataset", type=str, required=True,
                # choices = ['adult', 'german', 'compas', 'bank', 'mep'], help="Dataset name")
    # parser.add_argument("-p", "--protected", type=str, required=True,
                # help="Protected attribute")
    # 
    # args = parser.parse_args()
    # dataset_used = args.dataset
    # attr = args.protected
    ### Uncomment the code above to run dataset from terminal ###

    ################################# Data Info ######################################

    # dataset: adult        # protected attributes: sex, race 
    # dataset: german        # protected attributes: sex, age 
    # dataset: compas        # protected attributes: sex, race 
    # dataset: bank        # protected attributes: age
    # dataset: mep        # protected attributes: RACE 

    ##################################################################################

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


def read_data(dataset_name, seed) -> tuple[Dataset, Dataset, list, list]:

    dataset_orig, privileged_groups,unprivileged_groups, pos_class, label_name = get_data(dataset_name)
    df_train, df_test = train_test_split(dataset_orig, test_size=0.3, shuffle=True, random_state=seed)
    label_train = df_train[label_name].values
    label_test = df_test[label_name].values
    df_train = df_train.drop(columns=[label_name])
    df_test = df_test.drop(columns=[label_name]) 
    scaler = MinMaxScaler()
    scaler.fit(df_train)
    df_train = pd.DataFrame(scaler.transform(df_train), columns=df_train.columns)
    df_test = pd.DataFrame(scaler.transform(df_test), columns=df_test.columns)
    df_train[label_name] = label_train
    df_test[label_name] = label_test
    df_train = Dataset(df_train, label_name, pos_class, privileged_groups, unprivileged_groups)
    df_test = Dataset(df_test, label_name, pos_class, privileged_groups, unprivileged_groups)
    return df_train, df_test, privileged_groups, unprivileged_groups

    

def get_fairness_thresholds(dataset, attr=None, metric=None):
    threshold_df = pd.read_csv('./meg_fairness_thresholds.csv')
    filtered_exp = threshold_df[threshold_df['Exp'].str.contains(dataset, case = False) & threshold_df['Exp'].str.contains(attr, case = False)] if attr else threshold_df[threshold_df['Exp'].str.contains(dataset, case = False)]
    return abs(filtered_exp[metric].min())
