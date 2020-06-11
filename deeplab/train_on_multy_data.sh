#!/bin/bash

# Usage:
#   sh ./train_on_multy_data.sh 3 4 1,2,3,4
#   fist number 3 is used to choose server
#   second number 4 is used to set num_clones, it means we use 4 GPU to train
#   third parameter 1,2,3,4 means use GPU 1,2,3,4 in the sever

# Exit immediately if a command exits with a non-zero status.
set -e

cd ..

# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"

is_server=$1

datasets_type_name="MULTY_ORDER"
# set crop_size_dataset
if [ "$datasets_type_name" = "base" ]; then
    crop_size_dataset=289 #best 289
elif [ "$datasets_type_name" = "LTH" ]; then
    crop_size_dataset=257
elif [ "$datasets_type_name" = "HLJ" ]; then
    crop_size_dataset=289
elif [ "$datasets_type_name" = "SS" ]; then
    crop_size_dataset=273
elif [ "$datasets_type_name" = "MULTY" ]; then
    crop_size_dataset=305
elif [ "$datasets_type_name" = "FZLBJW" ]; then
    crop_size_dataset=305
elif [ "$datasets_type_name" = "HLJORDER" ]; then
    crop_size_dataset=289
elif [ "$datasets_type_name" = "SSORDER" ]; then
    crop_size_dataset=273
elif [ "$datasets_type_name" = "FZLBJWORDER" ]; then
    crop_size_dataset=305
elif [ "$datasets_type_name" = "LTHSKEL" ]; then
    crop_size_dataset=257
elif [ "$datasets_type_name" = "HLJSKEL" ]; then
    crop_size_dataset=289
elif [ "$datasets_type_name" = "SSSKEL" ]; then
    crop_size_dataset=273
elif [ "$datasets_type_name" = "MULTY_ORDER" ]; then
    crop_size_dataset=305
else
    crop_size_dataset=425
fi

datasets_type="DATA_GB6763_${datasets_type_name}"
DATA_YEAR_FOLDER="${datasets_type_name}2017"

# set DL_dataset_DIR base on the sever.
DL_dataset_DIR="/home/wwg/data/CCSSD"
if [ ${is_server} -eq 1 ]; then
    DL_dataset_DIR="/data/wangwg/CCSSD"
fi
if [ ${is_server} -eq 2 ]; then
    DL_dataset_DIR="/home/wangwg/CCSSD"
fi
if [ ${is_server} -eq 3 ]; then
    DL_dataset_DIR="/data1/wangwg/CCSSD"
fi
DATASET_DIR="${DL_dataset_DIR}/${datasets_type}/"

#cd "${WORK_DIR}/${DATASET_DIR}"
#sh download_and_convert_voc2012.sh

# Go back to original directory.
cd "${CURRENT_DIR}"

# Set up the working directories.

EXP_FOLDER="${DATASET_DIR}/${DATA_YEAR_FOLDER}/exp/train_on_trainval_set"
INIT_FOLDER="${DL_dataset_DIR}/init_models"

TRAIN_LOGDIR="${EXP_FOLDER}/train"
EVAL_LOGDIR="${EXP_FOLDER}/eval"
VIS_LOGDIR="${EXP_FOLDER}/vis"
EXPORT_DIR="${EXP_FOLDER}/export"
mkdir -p "${INIT_FOLDER}"
mkdir -p "${TRAIN_LOGDIR}"
mkdir -p "${EVAL_LOGDIR}"
mkdir -p "${VIS_LOGDIR}"
mkdir -p "${EXPORT_DIR}"



cd "${CURRENT_DIR}"

ZI_DATASET="${DATASET_DIR}/${DATA_YEAR_FOLDER}/tfrecord-cross-strokeOrder-addLTHlabel_double"

RUN_TRAIN=0
RUN_TEST=0
RUN_VIS=0
RUN_EXPO=0
if [ $1 -eq 0 ]; then
    RUN_VIS=1
else
    RUN_TRAIN=1
fi

fine_tune_stroke=0

checkpoint_select="${INIT_FOLDER}/xception/model.ckpt.index"

# Train 50000 iterations.
NUM_ITERATIONS=50000
if [ ${RUN_TRAIN} -eq 1 ]; then
  export CUDA_VISIBLE_DEVICES=$3
  python "${WORK_DIR}"/train.py \
    --logtostderr='/home/wwg/data1/models-master/research/deeplab' \
    --train_split="train" \
    --model_variant="xception_65" \
    --atrous_rates=3 \
    --atrous_rates=6 \
    --atrous_rates=9 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --train_crop_size=$crop_size_dataset \
    --train_crop_size=$crop_size_dataset \
    --train_batch_size=12 \
    --training_number_of_steps="${NUM_ITERATIONS}" \
    --fine_tune_batch_norm=true \
    --tf_initial_checkpoint="$checkpoint_select" \
    --train_logdir="${TRAIN_LOGDIR}" \
    --dataset_dir="${ZI_DATASET}" \
    --dataset="Zi"\
    --base_learning_rate=0.07 \
    --num_clones=$2 \
    --model_name="DeepStroke"
fi

