#!/bin/bash

for dataset in adult german compas bank mep;
do
  echo "Running for dataset: $dataset"
  python main.py --dataset $dataset
done