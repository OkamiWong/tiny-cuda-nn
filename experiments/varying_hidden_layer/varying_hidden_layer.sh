#!/bin/bash

SAFE_TEMPERATURE=55
COMPUTE_DEVICE_ID=1

function get_gpu_temperature() {
  local device_id=$1
  echo $(nvidia-smi -q -i $device_id -d TEMPERATURE | grep "GPU Current Temp" | grep -E "[0-9]+" -o)
}

function wait_for_safe_temperature() {
  current_temperature=$(get_gpu_temperature $COMPUTE_DEVICE_ID)
  while ((current_temperature > SAFE_TEMPERATURE)); do
    sleep 3
    current_temperature=$(get_gpu_temperature $COMPUTE_DEVICE_ID)
  done
}

BIN="../../build/mlp_learning_an_image"
INPUT="../../data/images/albert.jpg"
TINYCUDANN_CONFIG_TEMPLATE="./tinycudann-config.json"
MEMOPT_CONFIG_TEMPLATE="./memopt-config.json"
OUTPUT_FILE="./stdout.out"
HIDDEN_LAYER_NUMBERS=(4 5 6 7 8)

starting_directory=$(pwd)

for hidden_layer_number in ${HIDDEN_LAYER_NUMBERS[@]}; do
  wait_for_safe_temperature

  cd $starting_directory
  mkdir -p $hidden_layer_number
  cd $hidden_layer_number
  jq ".network.n_hidden_layers = $hidden_layer_number" ../$TINYCUDANN_CONFIG_TEMPLATE > $TINYCUDANN_CONFIG_TEMPLATE
  cp ../$MEMOPT_CONFIG_TEMPLATE ./config.json

  ../$BIN ../$INPUT $TINYCUDANN_CONFIG_TEMPLATE 100 &> $OUTPUT_FILE
done
