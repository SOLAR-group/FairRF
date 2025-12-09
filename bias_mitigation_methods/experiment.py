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
from aif360.algorithms.postprocessing import EqOddsPostprocessing

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

    ml_methods = {
        'logreg': LogisticRegression(),
        'svm': SVC(),
        'forest': RandomForestClassifier(),
        'knn': KNeighborsClassifier(),
        'cart': DecisionTreeClassifier()
    }
    ris = pd.DataFrame()
    for m in ml_methods.keys():
        model_fair = Pipeline([
            ('scaler', MinMaxScaler()),
            ('classifier', ml_methods[m])
        ])
        # model_fair = deepcopy(model)
        data = data.copy()
        metrics = deepcopy(base_metrics)
        train, test = train_test_split(data, test_size=0.3, random_state=run)
        X_train, X_test = train.drop(label, axis=1), test.drop(label, axis=1)
        y_train, y_test = train[label], test[label]
        instance_weights = None
        if fairness_methods == FairnessMethods.DEMV.value:
            print(f"Running DEMV for {m} on run {run}")
            dv = DEMV(sensitive_vars=sensitive_features)
            X_train, y_train = dv.fit_transform(X_train, y_train)

        elif fairness_methods == FairnessMethods.RW.value:
            print(f"Running Reweighing for {m} on run {run}")
            binary_data = BinaryLabelDataset(df=train, label_names=[label], protected_attribute_names=sensitive_features, favorable_label=positive_label)
            rw = Reweighing(unprivileged_groups=[unpriv_group], privileged_groups=[priv_group])
            train = rw.fit_transform(binary_data)
            X_train = train.features
            y_train = train.labels.ravel()
            instance_weights = train.instance_weights

        if not isinstance(model_fair.named_steps['classifier'], KNeighborsClassifier):
            model_fair.fit(X_train, y_train, classifier__sample_weight=instance_weights)
        else:
            model_fair.fit(X_train, y_train)

        y_pred = model_fair.predict(X_test)

        if fairness_methods == FairnessMethods.EOP.value:
            print(f"Running EOP for {m} on run {run}")
            
            binary_test_true = BinaryLabelDataset(df=test, label_names=[label], protected_attribute_names=sensitive_features, favorable_label=positive_label)
            binary_test_pred = BinaryLabelDataset(df=test, label_names=[label], protected_attribute_names=sensitive_features, favorable_label=positive_label)
            binary_test_pred.labels = y_pred.reshape(-1,1)
            eop = EqOddsPostprocessing(unprivileged_groups=[unpriv_group], privileged_groups=[priv_group])
            binary_test_pred = eop.fit_predict(binary_test_true, binary_test_pred)
            y_pred = binary_test_pred.labels.ravel()

        test_df = test.copy()
        test_df['y_true'] = y_test
        test_df[label] = y_pred
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
