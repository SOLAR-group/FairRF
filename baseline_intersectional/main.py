import pandas as pd
import argparse
import os
import pickle
from mydatautil import get_data

from experiment import run_exp

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Experiment file for fairness testing')
    parser.add_argument('-d', '--dataset', type=str,help='Required argument: relative path of the dataset to process')

    args = parser.parse_args()
    
    for run in range(20):
        data, privileged_groups, unprivileged_groups, pos_class, label_name = get_data(args.dataset)
        os.makedirs(f"results/{args.dataset}", exist_ok=True)
        run_exp(data, label=label_name, 
                                positive_label=pos_class, 
                                unpriv_group=unprivileged_groups, 
                                priv_group=privileged_groups, 
                                data_name=f"results/{args.dataset}/round_{run}.csv",
                                run=run)
