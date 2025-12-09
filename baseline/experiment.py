import os
import pickle
from copy import deepcopy
import utils
from datetime import datetime
from methods import FairnessMethods
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

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



def run_exp(data, label, positive_label=1, unpriv_group=None, priv_group=None, data_name=None, run=0):
    sensitive_features = unpriv_group.keys()
    save_data =  True   
    save_model = False
    ml_methods = {
        'logreg': LogisticRegression(),
        'svm': SVC(),
        'forest': RandomForestClassifier(),
        # 'knn' : KNeighborsClassifier(),
        # 'tree': DecisionTreeClassifier(),
    }

    fairness_methods = {
        'no_method': FairnessMethods.NO_ONE,
    }
    ris = pd.DataFrame()
    for m in ml_methods.keys():
        model = Pipeline([
            ('scaler', MinMaxScaler()),
            ('classifier', ml_methods[m])
        ])
        for f in fairness_methods.keys():
            model = deepcopy(model)
            data = data.copy()
            # if f == 'preprocessing':
            #     for method in fairness_methods[f]:
            #         metrics = deepcopy(base_metrics)
            #         model_fair, ris_metrics = cross_val(classifier=model, data=data, unpriv_group=unpriv_group, priv_group=priv_group, label=label, metrics=metrics, positive_label=positive_label, sensitive_features=sensitive_features, preprocessor=method, n_splits=30)
            #         df_metrics = _store_metrics(ris_metrics, m, method.name, save_data, save_model, model_fair)
            #         ris = ris.append(df_metrics)
            # elif f == 'inprocessing':
            #     for method in fairness_methods[f]:
            #         metrics = deepcopy(base_metrics)
            #         model_fair, ris_metrics = cross_val(classifier=model, data=data, unpriv_group=unpriv_group, priv_group=priv_group, label=label, metrics=metrics, positive_label=positive_label, sensitive_features=sensitive_features, inprocessor=method, n_splits=30)
            #         df_metrics = _store_metrics(
            #             ris_metrics, m, method.name, save_data, save_model, model_fair)
            #         ris = ris.append(df_metrics)
            # elif f == 'postprocessing':
            #     for method in fairness_methods[f]:
            #         metrics = deepcopy(base_metrics)
            #         model_fair, ris_metrics = cross_val(classifier=model, data=data, unpriv_group=unpriv_group, priv_group=priv_group, label=label, metrics=metrics, positive_label=positive_label,sensitive_features=sensitive_features, postprocessor=method, n_splits=30)
            #         df_metrics = _store_metrics(ris_metrics, m, method.name, save_data, save_model, model_fair)
            #         ris = ris.append(df_metrics)
            # else:
            metrics = deepcopy(base_metrics)



            # model_fair, ris_metrics = cross_val(classifier=model, data=data, unpriv_group=unpriv_group, priv_group=priv_group, label=label, metrics=metrics, positive_label=positive_label,sensitive_features=sensitive_features, n_splits=30)
            #for train, test in KFold(n_splits=2, shuffle=True, random_state=run).split(data):
            train, test = train_test_split(data, test_size=0.3, random_state=run)
            model_fair = deepcopy(model)
            X_train, X_test = train.drop(label, axis=1), test.drop(label, axis=1)
            y_train, y_test = train[label], test[label]
            # X_train, X_test = data.iloc[train].drop(label, axis=1), data.iloc[test].drop(label, axis=1)
            # y_train, y_test = data[label].iloc[train], data[label].iloc[test]
            model_fair.fit(X_train, y_train)
            y_pred = model_fair.predict(X_test)
            test_df = test.copy()
            test_df['y_true'] = y_test
            test_df[label] = y_pred
            metrics = utils.compute_metrics(test_df, unpriv_group=unpriv_group, label=label, positive_label=positive_label, metrics=metrics, sensitive_features=sensitive_features)
            ris_metrics = deepcopy(metrics)

            df_metrics = _store_metrics(ris_metrics, m, FairnessMethods.NO_ONE.name, save_data, save_model, model_fair, data_name)
            ris = pd.concat([ris, df_metrics], ignore_index=True)
    if save_data:
        ris_old = pd.read_csv(data_name) if os.path.exists(data_name) else pd.DataFrame()
        ris = pd.concat([ris_old, ris], ignore_index=True)
        ris.to_csv(data_name, index=False)
    

