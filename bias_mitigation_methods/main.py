import pandas as pd
import argparse
import os
import pickle
from mydatautil import get_data
from methods import FairnessMethods

from experiment import run_exp
from experiment_adv import run_exp as run_exp_adv

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Experiment file for fairness testing')
    parser.add_argument('-d', '--dataset', type=str,help='Required argument: relative path of the dataset to process')
    parser.add_argument('-p', '--protected', type=str,help='Required argument: relative path of the dataset to process')
    parser.add_argument('-m', '--method', default=FairnessMethods.NO_ONE.value)

    args = parser.parse_args()
    
    for run in range(20):
        data, privileged_groups, unprivileged_groups, pos_class, label_name = get_data(args.dataset, args.protected)
        os.makedirs(f"results_{args.method}/{args.dataset}/{args.protected}", exist_ok=True)
        if args.method == FairnessMethods.AD.value:
            run_exp_adv(data, label=label_name, 
                                positive_label=pos_class, 
                                unpriv_group=unprivileged_groups, 
                                priv_group=privileged_groups, 
                                data_name=f"results_{args.method}/{args.dataset}/{args.protected}/round_{run}.csv",
                                run=run, 
                                fairness_methods=args.method)
        else:
            run_exp(data, label=label_name, 
                                positive_label=pos_class, 
                                unpriv_group=unprivileged_groups, 
                                priv_group=privileged_groups, 
                                data_name=f"results_{args.method}/{args.dataset}/{args.protected}/round_{run}.csv",
                                run=run, 
                                fairness_methods=args.method)
