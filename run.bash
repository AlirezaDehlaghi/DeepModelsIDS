#!/bin/bash

# Activate the virtual environment
source myenv/bin/activate

models=("AutoEncoder" "VariationalAutoEncoder" "ConditionalVariationalAutoEncoder" "LSTMAutoEncoder" "ProbabilisticVariationalEncoder")

for i in {1..10}
do
  echo "Iteration $i"

  # Training
  for model in "${models[@]}"
  do
    echo "Training $model"
    python ResourceProfiler.py train "$model"
    echo "Cooling down after training $model"
    sleep 10
  done

  # Testing
  for model in "${models[@]}"
  do
    echo "Testing $model"
    python ResourceProfiler.py test "$model"
    echo "Cooling down after testing $model"
    sleep 10
  done
done