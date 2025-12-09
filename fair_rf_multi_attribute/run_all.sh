#!/bin/bash

for dataset in adult german compas;
do
  echo "Running for dataset: $dataset"
  python main.py --dataset $dataset -r 0
done