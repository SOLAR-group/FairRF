import os
import pickle
from copy import deepcopy
import utils
from datetime import datetime
from methods import FairnessMethods
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from demv import DEMV
from aif360.algorithms.preprocessing import Reweighing
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.inprocessing import AdversarialDebiasing
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from utils import *


base_metrics = {
    'precision': [],
    'recall': [],
    'f1score': [],
    'mcc': [],
    'stat_par': [],
    'eq_odds': [],
    'ao': [],
    'acc': [],
}

def _store_metrics(metrics, method, fairness, save_data, save_model, model_fair, data_name):
    df_metrics = pd.DataFrame(metrics)
    df_metrics = df_metrics.explode(list(df_metrics.columns))
    df_metrics['model'] = method
    df_metrics['fairness_method'] = fairness
    # if save_data:
    #     df_metrics.to_csv(data_name, index=False)
    return df_metrics



def run_exp(data, label, positive_label=1, unpriv_group=None, priv_group=None, data_name=None, run=0, fairness_methods=FairnessMethods.NO_ONE):
    sensitive_features = list(unpriv_group.keys())
    save_data =  True   
    save_model = False
    tf.reset_default_graph()
    sess = tf.Session()
    ml_methods = {
        'adversarial_debiasing': AdversarialDebiasing(privileged_groups=[priv_group],
                                                        unprivileged_groups=[unpriv_group],
                                                        scope_name='debiased_classifier',
                                                        debias=True,
                                                        sess=sess)
        }

    ris = pd.DataFrame()
    for m in ml_methods.keys():
        model_fair = ml_methods[m]
        # model_fair = deepcopy(model)
        data = data.copy()
        metrics = deepcopy(base_metrics)
        train, test = train_test_split(data, test_size=0.3, random_state=run)
        X_train, X_test = train.drop(label, axis=1), test.drop(label, axis=1)
        y_train, y_test = train[label], test[label]
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        binary_data_train = BinaryLabelDataset(df=train, label_names=[label], protected_attribute_names=sensitive_features, favorable_label=positive_label)
        binary_data_test = BinaryLabelDataset(df=test, label_names=[label], protected_attribute_names=sensitive_features, favorable_label=positive_label)

        # model_fair = deepcopy(model)

        # X_train, X_test = data.iloc[train].drop(label, axis=1), data.iloc[test].drop(label, axis=1)
        # y_train, y_test = data[label].iloc[train], data[label].iloc[test]
        # try:
        model_fair.fit(binary_data_train)
        bin_data_pred = model_fair.predict(binary_data_test)
        test_df = binary_data_test.convert_to_dataframe()[0]
        test_df['y_true'] = y_test
        test_df[label] = bin_data_pred.labels
        metrics = utils.compute_metrics(test_df, unpriv_group=unpriv_group, label=label, positive_label=positive_label, metrics=metrics, sensitive_features=sensitive_features)
        ris_metrics = deepcopy(metrics)

        df_metrics = _store_metrics(ris_metrics, m, FairnessMethods.NO_ONE.name, save_data, save_model, model_fair, data_name)
        ris = pd.concat([ris, df_metrics], ignore_index=True)
        if save_data:
            ris.to_csv(data_name, index=False)
        # except Exception as e:
        #     print(f"Error occurred while saving data for {m} on run {run}: {e}")
        #     print(e.args)
        #     continue
