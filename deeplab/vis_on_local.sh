#!/bin/bash
# Usage:
#   sh ./vis_on_local.sh
#

# Exit immediately if a command exits with a non-zero status.
set -e

cd ..

# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"

# if want to get result of handwritten data, you can add "hand" behind the "FZLBJW"
for datasets_type_name in "HLJ" "SS" "FZLBJW"
#"HLJ" "SS" "FZLBJW"
#'hand'

do
    # set crop_size_dataset
    echo $datasets_type_name
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

    DL_dataset_DIR="/home/wwg/data/CCSSD"
    DATASET_DIR="${DL_dataset_DIR}/${datasets_type}/"

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

    ## add TRAIN_LOGDIR
    TRAIN_LOGDIR="/home/wwg/data/CCSSD/DATA_GB6763_MULTY_ORDER/MULTY_ORDER2017/exp/train_on_trainval_set/train"
    ZI_DATASET="${DATASET_DIR}/${DATA_YEAR_FOLDER}/tfrecord-cross-strokeOrder-addLTHlabel_double"

    RUN_VIS=1

    checkpoint_select="${INIT_FOLDER}/xception/model.ckpt.index"

    # Visualize the results.
    if [ ${RUN_VIS} -eq 1 ]; then
      python "${WORK_DIR}"/vis.py \
        --logtostderr \
        --vis_split="trainval" \
        --model_variant="xception_65" \
        --atrous_rates=3 \
        --atrous_rates=6 \
        --atrous_rates=9 \
        --output_stride=16 \
        --decoder_output_stride=4 \
        --vis_crop_size=$crop_size_dataset \
        --vis_crop_size=$crop_size_dataset \
        --checkpoint_dir="${TRAIN_LOGDIR}"\
        --vis_logdir="${VIS_LOGDIR}" \
        --dataset_dir="${ZI_DATASET}" \
        --max_number_of_iterations=1 \
        --dataset="Zi" \
        --also_save_raw_predictions=true \
        --model_name="DeepStroke"
    fi
done