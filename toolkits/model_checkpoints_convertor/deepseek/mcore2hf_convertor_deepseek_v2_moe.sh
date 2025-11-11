#!/bin/bash
set -e
START_TIME=$SECONDS

conda_cmd="source activate && conda activate python310_torch25_cuda"
echo ${conda_cmd}
eval ${conda_cmd} 

ROOT_DIR=$( dirname -- "$( readlink -f -- "$0"; )"; )
ROOT_DIR=${ROOT_DIR}/../../..

source_cmd="source ${ROOT_DIR}/zj_examples/V5000/xpu_env.sh"
echo ${source_cmd}
eval ${source_cmd} 
unset USE_FAST_BF16_FC

export PYTHONPATH=${PYTHONPATH}:${ROOT_DIR}/Megatron-LM/${MCORE_VERSION:-core_r0_10_0}:${ROOT_DIR}/megatron_patch/${MEGATRON_PATCH_VERSION:-megatron_patch_250328}
export CUDA_DEVICE_MAX_CONNECTIONS=1

MP_PP0_LAYERS=${MP_PP0_LAYERS:-}
TP=${TP:-1}
PP=${PP:-1}
EP=${EP:-4}

MODEL_SIZE=A2.4B
HF_CKPT_PATH=${ROOT_DIR}/../model/DeepSeek-V2-Lite
SOURCE_CKPT_PATH=${ROOT_DIR}/../ckpt/DeepSeek-V2-Lite-mcore-TP-${TP}-PP-${PP}-EP-${EP}
TARGET_CKPT_PATH=${ROOT_DIR}/../ckpt/DeepSeek-V2-Lite-hf

PR=bf16
TE=true
MG2HF=true

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-9988}

DISTRIBUTED_ARGS="--nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

if [ $MODEL_SIZE = A2.4B ]; then

HIDDEN_SIZE=2048
NUM_ATTENTION_HEADS=16
NUM_LAYERS=27
INTERMEDIATE_SIZE=10944
MOE_INTERMEDIATE_SIZE=1408
EXTRA_VOCAB_SIZE=2400
KV_LORA_RANK=512
QK_NOPE_HEAD_DIM=128
QK_ROPE_HEAD_DIM=64
V_HEAD_DIM=128
ROPE_THETA=10000
SCALE_FACTOR=40
NUM_EXPERTS=64
ROUTER_TOPK=6
NUM_SHARED_EXPERTS=2
MOE_LAYER_FREQ=1
RMS_NORM_EPS=1e-6

moe_options=" \
    --moe-ffn-hidden-size ${MOE_INTERMEDIATE_SIZE} \
    --moe-router-topk ${ROUTER_TOPK} \
    --num-experts ${NUM_EXPERTS} \
    --moe-layer-freq ${MOE_LAYER_FREQ} \
    --moe-aux-loss-coeff 1e-2 \
    --moe-shared-expert-intermediate-size $((${MOE_INTERMEDIATE_SIZE} * ${NUM_SHARED_EXPERTS} )) \
    --target-expert-model-parallel-size ${EP} \
    --kv-lora-rank ${KV_LORA_RANK} \
    --qk-head-dim ${QK_NOPE_HEAD_DIM} \
    --qk-pos-emb-head-dim ${QK_ROPE_HEAD_DIM} \
    --v-head-dim ${V_HEAD_DIM} \
    --moe-router-load-balancing-type aux_loss"

cpu_options=" \
            --use-cpu-initialization"

elif [ $MODEL_SIZE = A21B ]; then

HIDDEN_SIZE=5120
NUM_ATTENTION_HEADS=128
NUM_LAYERS=60
INTERMEDIATE_SIZE=12288
MOE_INTERMEDIATE_SIZE=1536
EXTRA_VOCAB_SIZE=2400
Q_LORA_RANK=1536
KV_LORA_RANK=512
QK_NOPE_HEAD_DIM=128
QK_ROPE_HEAD_DIM=64
V_HEAD_DIM=128
ROPE_THETA=10000
SCALE_FACTOR=40
NUM_EXPERTS=160
ROUTER_TOPK=6
NUM_SHARED_EXPERTS=2
MOE_LAYER_FREQ=1
RMS_NORM_EPS=1e-6

moe_options=" \
    --moe-router-topk ${ROUTER_TOPK} \
    --num-experts ${NUM_EXPERTS} \
    --target-expert-model-parallel-size ${EP} \
    --moe-ffn-hidden-size ${MOE_INTERMEDIATE_SIZE} \
    --moe-router-load-balancing-type aux_loss \
    --moe-aux-loss-coeff 1e-2 \
    --enable-shared-expert \
    --moe-layer-freq ${MOE_LAYER_FREQ} \
    --moe-shared-expert-intermediate-size $((${MOE_INTERMEDIATE_SIZE} * ${NUM_SHARED_EXPERTS} )) \
    --q-lora-rank ${Q_LORA_RANK} \
    --kv-lora-rank ${KV_LORA_RANK} \
    --qk-nope-head-dim ${QK_NOPE_HEAD_DIM} \
    --qk-rope-head-dim ${QK_ROPE_HEAD_DIM} \
    --v-head-dim ${V_HEAD_DIM} \
    "

cpu_options=" \
            --use-cpu-initialization"

fi


if [ $MG2HF = true ]; then
    convert_options=" \
                --convert-checkpoint-from-megatron-to-transformers \
                --hf-ckpt-path ${HF_CKPT_PATH}"

elif [ $MG2HF = false ]; then
    convert_options=""
fi

if [ $PR = fp16 ]; then
    pr_options=" \
		    --fp16"

elif [ $PR = bf16 ]; then
    pr_options=" \
        --bf16"

fi

if [ -z ${MP_PP0_LAYERS} ];then
    uneven_split_option=""
elif [ ${PP} -gt 1 ]; then
    _check=$(( ( $NUM_LAYERS - ${MP_PP0_LAYERS} ) % ( ${PP} - 1 ) ))
    if [ $_check != 0 ]; then
        echo "With uneven pipelineing the left over layers must be divisible by left over stages."
        exit -1
    fi

    uneven_split_option=" \
        --target-decoder-first-pipeline-num-layers ${MP_PP0_LAYERS}
    "
else
    echo "uneven pipeline split must be used when PP > 1"
    exit -1
fi

if [ $TE = true ]; then
    te_options=" \
		    --transformer-impl transformer_engine"

elif [ $TE = false ]; then
    te_options=" \
        --transformer-impl local"
fi

torchrun ${DISTRIBUTED_ARGS} hf2mcore_deepseek_v2_moe.py \
    --load ${SOURCE_CKPT_PATH} \
    --save ${TARGET_CKPT_PATH} \
    --target-tensor-model-parallel-size ${TP} \
    --target-pipeline-model-parallel-size ${PP} \
    --micro-batch-size 1 \
    --save-interval 1 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --swiglu \
    --num-layers ${NUM_LAYERS} \
    --hidden-size ${HIDDEN_SIZE} \
    --ffn-hidden-size ${INTERMEDIATE_SIZE} \
    --num-attention-heads ${NUM_ATTENTION_HEADS} \
    --max-position-embeddings 10 \
    --max-padding-length 10 \
    --seq-length 10 \
    --no-async-tensor-model-parallel-allreduce \
    --patch-tokenizer-type DeepSeekV2Tokenizer \
    --extra-vocab-size ${EXTRA_VOCAB_SIZE} \
    --untie-embeddings-and-output-weights \
    --no-bias-swiglu-fusion \
    --no-rope-fusion \
    --position-embedding-type rope \
    --use-rotary-position-embeddings \
    --disable-bias-linear \
    --normalization RMSNorm \
    --norm-epsilon ${RMS_NORM_EPS} \
    --rotary-base ${ROPE_THETA} \
    --rotary-scaling-factor ${SCALE_FACTOR} \
    --rotary-seq-len-interpolation-factor 1 \
    --kv-channels ${V_HEAD_DIM} \
    --qk-layernorm \
    --multi-latent-attention \
    ${te_options} \
    ${moe_options} \
    ${convert_options} \
    ${pr_options} \
    ${uneven_split_option} \
    ${cpu_options}

ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "$(($ELAPSED_TIME/60)) min $(($ELAPSED_TIME%60)) sec"