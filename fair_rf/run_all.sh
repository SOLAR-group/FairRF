#!/bin/bash

for dataset in adult german compas bank mep;
do
  for model in rf knn cart lr svm; do
    echo "Running for dataset: $dataset"
    python main.py --dataset $dataset --model $model
done
done