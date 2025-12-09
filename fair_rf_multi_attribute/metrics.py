import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
)


class Metrics:
    def __init__(self, data: pd.DataFrame, label_name: str, true_label: str, positive_label: int):
        """
        :param data: DataFrame containing the predictions and true labels
        :param label_name: Name of the column containing predicted labels
        :param true_label: Name of the column containing true labels
        :param positive_label: The label considered as positive (e.g., 1 for binary classification)
        """

        self.data_pred = data
        self.label_name = label_name
        self.positive_label = positive_label
        self.true_label = true_label

    def __get_groups(self, group_condition, predicted=True):
        query = "&".join(
            [f"`{str(k)}`" + "==" + str(v) for k, v in group_condition.items()]
        )
        label_query = (
            f"`{self.label_name}`" + "==" + str(self.positive_label)
            if predicted
            else f"`{self.true_label}`" + "==" + str(self.positive_label)
        )
        unpriv_group = self.data_pred.query(query)
        unpriv_group_pos = self.data_pred.query(query + "&" + label_query)
        priv_group = self.data_pred.query("~(" + query + ")")
        priv_group_pos = self.data_pred.query("~(" + query + ")&" + label_query)
        return unpriv_group, unpriv_group_pos, priv_group, priv_group_pos

    def compute_probs(self, group_condition, predicted=True):
        """
        Computes the probabilities of the positive label in the unprivileged and privileged groups.
        :param group_condition: Dictionary defining the group condition
        """
        unpriv_group, unpriv_group_pos, priv_group, priv_group_pos = self.__get_groups(
            group_condition, predicted
        )
        try:
            unpriv_group_prob = len(unpriv_group_pos) / len(unpriv_group)
        except:
            unpriv_group_prob = 0
        try:
            priv_group_prob = len(priv_group_pos) / len(priv_group)
        except:
            priv_group_prob = 0
        return unpriv_group_prob, priv_group_prob

    def __compute_tpr_fpr(self, y_true, y_pred):
        TN = 0
        TP = 0
        FP = 0
        FN = 0
        for i in range(len(y_true)):
            if y_true[i] == self.positive_label:
                if y_true[i] == y_pred[i]:
                    TP += 1
                else:
                    FN += 1
            else:
                if y_pred[i] == self.positive_label:
                    FP += 1
                else:
                    TN += 1
        if TP + FN == 0:
            TPR = 0
        else:
            TPR = TP / (TP + FN)
        if FP + TN == 0:
            FPR = 0
        else:
            FPR = FP / (FP + TN)
        return FPR, TPR

    def __compute_tpr_fpr_groups(self, group_condition):
        query = "&".join([f"`{k}`=={v}" for k, v in group_condition.items()])
        unpriv_group = self.data_pred.query(query)
        priv_group = self.data_pred.drop(unpriv_group.index)

        y_true_unpriv = unpriv_group[self.true_label].values.ravel()
        y_pred_unpric = unpriv_group[self.label_name].values.ravel()
        y_true_priv = priv_group[self.true_label].values.ravel()
        y_pred_priv = priv_group[self.label_name].values.ravel()

        fpr_unpriv, tpr_unpriv = self.__compute_tpr_fpr(y_true_unpriv, y_pred_unpric)
        fpr_priv, tpr_priv = self.__compute_tpr_fpr(y_true_priv, y_pred_priv)
        return fpr_unpriv, tpr_unpriv, fpr_priv, tpr_priv

    def group_ratio(self, group_condition):
        unpriv_group, unpriv_group_pos, priv_group, priv_group_pos = self.__get_groups(
            group_condition, False
        )
        unpriv_ratio = 0
        priv_ratio = 0
        if len(unpriv_group_pos) > 0:
            w_exp = (len(unpriv_group) / len(self.data_pred)) * (
                len(
                    self.data_pred[
                        self.data_pred[self.true_label] == self.positive_label
                    ]
                )
                / len(self.data_pred)
            )
            w_obs = len(unpriv_group_pos) / len(self.data_pred)
            unpriv_ratio = w_obs / w_exp
        if len(priv_group_pos) > 0:
            w_exp = (len(priv_group) / len(self.data_pred)) * (
                len(
                    self.data_pred[
                        self.data_pred[self.true_label] == self.positive_label
                    ]
                )
                / len(self.data_pred)
            )
            w_obs = len(priv_group_pos) / len(self.data_pred)
            priv_ratio = w_obs / w_exp
        return unpriv_ratio, priv_ratio

    def disparate_impact(self, group_condition):
        unpriv_group_prob, priv_group_prob = self.compute_probs(group_condition)
        return (
            min(
                unpriv_group_prob / priv_group_prob, priv_group_prob / unpriv_group_prob
            )
            if unpriv_group_prob != 0 and priv_group_prob != 0
            else 0
        )

    def statistical_parity(self, group_condition: dict):
        unpriv_group_prob, priv_group_prob = self.compute_probs(group_condition)
        return unpriv_group_prob - priv_group_prob

    def disparate_impact(self, group_condition: dict):
        unpriv_group_prob, priv_group_prob = self.compute_probs(group_condition)
        return unpriv_group_prob / priv_group_prob if priv_group_prob != 0 else 0

    def average_odds(self, group_condition: dict):
        fpr_unpriv, tpr_unpriv, fpr_priv, tpr_priv = self.__compute_tpr_fpr_groups(
            group_condition
        )
        return ((tpr_unpriv - tpr_priv) + (fpr_unpriv - fpr_priv)) / 2

    def equal_opportunity(self, group_condition: dict):
        fpr_unpriv, tpr_unpriv, fpr_priv, tpr_priv = self.__compute_tpr_fpr_groups(
            group_condition
        )
        return tpr_unpriv - tpr_priv


    def wc_spd(self, p_attrs):
        attr1 = p_attrs[0]
        attr2 = p_attrs[1]
        num1 = len(self.data_pred[(self.data_pred[attr1] == 0) & (self.data_pred[attr2] == 0) & (self.data_pred[self.label_name] == 1)]) / len(
            self.data_pred[(self.data_pred[attr1] == 0) & (self.data_pred[attr2] == 0)])
        num2 = len(self.data_pred[(self.data_pred[attr1] == 0) & (self.data_pred[attr2] == 1) & (self.data_pred[self.label_name] == 1)]) / len(
            self.data_pred[(self.data_pred[attr1] == 0) & (self.data_pred[attr2] == 1)])
        num3 = len(self.data_pred[(self.data_pred[attr1] == 1) & (self.data_pred[attr2] == 0) & (self.data_pred[self.label_name] == 1)]) / len(
            self.data_pred[(self.data_pred[attr1] == 1) & (self.data_pred[attr2] == 0)])
        num4 = len(self.data_pred[(self.data_pred[attr1] == 1) & (self.data_pred[attr2] == 1) & (self.data_pred[self.label_name] == 1)]) / len(
            self.data_pred[(self.data_pred[attr1] == 1) & (self.data_pred[attr2] == 1)])
        return [num1, num2, num3, num4, max([num1,num2,num3,num4])-min([num1,num2,num3,num4])]

    def wc_aod(self, p_attrs):
        attr1 = p_attrs[0]
        attr2 = p_attrs[1]
        # favorlabel = dataset_test.favorable_label
        # labelname = dataset_test.label_names[0]
        # dataset_test = dataset_test.convert_to_dataframe()[0]
        # dataset_test_pred = dataset_test_pred.convert_to_dataframe()[0]
        # dataset_test['pred'+labelname] = dataset_test_pred[labelname]
        # dataset_test[labelname] = np.where(dataset_test[labelname] == favorlabel, 1, 0)
        # dataset_test['pred'+labelname] = np.where(dataset_test['pred'+labelname] == favorlabel, 1, 0)
        num_list = []
        # if len(dataset_test[(dataset_test[attr1] == 0) & (dataset_test[attr2] == 0) & (dataset_test[labelname] == 0)]) != 0 and len(
        #         dataset_test[(dataset_test[attr1] == 0) & (dataset_test[attr2] == 0) & (dataset_test[
        #             labelname] == 1)]) !=0:
        num1 = len(self.data_pred[(
                self.data_pred[attr1] == 0) & (self.data_pred[attr2] == 0) & (self.data_pred[
            self.true_label] == 0) & (self.data_pred[self.label_name] == 1)]) / len(
            self.data_pred[(self.data_pred[attr1] == 0) & (self.data_pred[attr2] == 0) & (self.data_pred[
                self.true_label] == 0)]) + len(self.data_pred[(
                self.data_pred[attr1] == 0) & (self.data_pred[attr2] == 0) & (self.data_pred[
            self.true_label] == 1) & (self.data_pred[self.label_name] == 1)]) / len(
            self.data_pred[(self.data_pred[attr1] == 0) & (self.data_pred[attr2] == 0) & (self.data_pred[
                self.true_label] == 1)])
        num_list.append(num1)
        # if len(dataset_test[(dataset_test[attr1] == 0) & (dataset_test[attr2] == 1) & (dataset_test[labelname] == 0)]) != 0 and len(
        #         dataset_test[(dataset_test[attr1] == 0) & (dataset_test[attr2] == 1) & (dataset_test[
        #             labelname] == 1)])!=0:
        num2 = len(self.data_pred[(
                self.data_pred[attr1] == 0) & (self.data_pred[attr2] == 1) & (self.data_pred[
            self.true_label] == 0) & (self.data_pred[self.label_name] == 1)]) / len(
            self.data_pred[(self.data_pred[attr1] == 0) & (self.data_pred[attr2] == 1) & (self.data_pred[
                self.true_label] == 0)]) + len(self.data_pred[(
                self.data_pred[attr1] == 0) & (self.data_pred[attr2] == 1) & (self.data_pred[
            self.true_label] == 1) & (self.data_pred[self.label_name] == 1)]) / len(
            self.data_pred[(self.data_pred[attr1] == 0) & (self.data_pred[attr2] == 1) & (self.data_pred[
                self.true_label] == 1)])
        num_list.append(num2)
        # if len(dataset_test[(dataset_test[attr1] == 1) & (dataset_test[attr2] == 0) & (dataset_test[labelname] == 0)]) != 0 and len(
        #         dataset_test[(dataset_test[attr1] == 1) & (dataset_test[attr2] == 0) & (dataset_test[
        #             labelname] == 1)])!=0:
        num3 = len(self.data_pred[(
                self.data_pred[attr1] == 1) & (self.data_pred[attr2] == 0) & (self.data_pred[
            self.true_label] == 0) & (self.data_pred[self.label_name] == 1)]) / len(
            self.data_pred[(self.data_pred[attr1] == 1) & (self.data_pred[attr2] == 0) & (self.data_pred[
                self.true_label] == 0)]) + len(self.data_pred[(
                self.data_pred[attr1] == 1) & (self.data_pred[attr2] == 0) & (self.data_pred[
            self.true_label] == 1) & (self.data_pred[self.label_name] == 1)]) / len(
            self.data_pred[(self.data_pred[attr1] == 1) & (self.data_pred[attr2] == 0) & (self.data_pred[
                self.true_label] == 1)])
        num_list.append(num3)
        # if len(dataset_test[(dataset_test[attr1] == 1) & (dataset_test[attr2] == 1) & (dataset_test[labelname] == 0)]) != 0 and len(
        #         dataset_test[(dataset_test[attr1] == 1) & (dataset_test[attr2] == 1) & (dataset_test[
        #             labelname] == 1)]) != 0:
        num4 = len(self.data_pred[(
                self.data_pred[attr1] == 1) & (self.data_pred[attr2] == 1) & (self.data_pred[
            self.true_label] == 0) & (self.data_pred[self.label_name] == 1)]) / len(
            self.data_pred[(self.data_pred[attr1] == 1) & (self.data_pred[attr2] == 1) & (self.data_pred[
                self.true_label] == 0)]) + len(self.data_pred[(
                self.data_pred[attr1] == 1) & (self.data_pred[attr2] == 1) & (self.data_pred[
            self.true_label] == 1) & (self.data_pred[self.label_name] == 1)]) / len(
            self.data_pred[(self.data_pred[attr1] == 1) & (self.data_pred[attr2] == 1) & (self.data_pred[
                self.true_label] == 1)])
        num_list.append(num4)
        return [num1, num2, num3, num4, (max(num_list) - min(num_list))/2]

    def wc_eod(self, p_attrs):
        attr1 = p_attrs[0]
        attr2 = p_attrs[1]
        # favorlabel = 1
        # labelname = 'Probability'
        # dataset_test = dataset_test.convert_to_dataframe()[0]
        # dataset_test_pred = dataset_test_pred.convert_to_dataframe()[0]
        # dataset_test['pred'+labelname] = dataset_test_pred[labelname]
        # dataset_test[labelname] = np.where(dataset_test[labelname] == favorlabel, 1, 0)
        # dataset_test[self.label_name] = np.where(dataset_test[self.label_name] == favorlabel, 1, 0)
        num_list=[]
        # if len(dataset_test[(dataset_test[attr1] == 0) & (dataset_test[attr2] == 0) & (dataset_test[labelname] == 1)]) != 0:
        num1 = len(self.data_pred[(
                self.data_pred[attr1] == 0) & (self.data_pred[attr2] == 0) & (self.data_pred[
            self.true_label] == 1) & (self.data_pred[self.label_name] == 1)]) / len(
            self.data_pred[(self.data_pred[attr1] == 0) & (self.data_pred[attr2] == 0) & (self.data_pred[
            self.true_label] == 1)])
        num_list.append(num1)
        # if len(dataset_test[(dataset_test[attr1] == 0) & (dataset_test[attr2] == 1) & (dataset_test[labelname] == 1)]) != 0:
        num2 = len(self.data_pred[(
                self.data_pred[attr1] == 0) & (self.data_pred[attr2] == 1) & (self.data_pred[
            self.true_label] == 1) & (self.data_pred[self.label_name] == 1)]) / len(
            self.data_pred[(self.data_pred[attr1] == 0) & (self.data_pred[attr2] == 1) & (self.data_pred[
                self.true_label] == 1)])
        num_list.append(num2)
        # if len(dataset_test[(dataset_test[attr1] == 1) & (dataset_test[attr2] == 0) & (dataset_test[labelname] == 1)]) != 0:
        num3 = len(self.data_pred[(
                self.data_pred[attr1] == 1) & (self.data_pred[attr2] == 0) & (self.data_pred[
            self.true_label] == 1) & (self.data_pred[self.label_name] == 1)]) / len(
            self.data_pred[(self.data_pred[attr1] == 1) & (self.data_pred[attr2] == 0) & (self.data_pred[
                self.true_label] == 1)])
        num_list.append(num3)
        # if len(dataset_test[(dataset_test[attr1] == 1) & (dataset_test[attr2] == 1) & (dataset_test[labelname] == 1)]) != 0:
        num4 = len(self.data_pred[(
                self.data_pred[attr1] == 1) & (self.data_pred[attr2] == 1) & (self.data_pred[
            self.true_label] == 1) & (self.data_pred[self.label_name] == 1)]) / len(
            self.data_pred[(self.data_pred[attr1] == 1) & (self.data_pred[attr2] == 1) & (self.data_pred[
                self.true_label] == 1)])
        num_list.append(num4)
        return [num1, num2, num3, num4, max(num_list) - min(num_list)]
    