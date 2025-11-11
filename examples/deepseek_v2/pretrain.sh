#!/bin/bash
set -exo pipefail 

ipcs -m | awk '$4 == 666 {print $2}' | while read shmid; do
    ipcrm -m $shmid
    echo "Deleted shared memory segment with ID: $shmid"
done  

conda_cmd="source activate && conda activate python310_torch25_cuda"
echo ${conda_cmd}
eval ${conda_cmd}

source ./V5000/xpu_env.sh
ENV=dsw
ROOT_DIR=$( dirname -- "$( readlink -f -- "$0"; )"; )
ROOT_DIR=$(realpath ${ROOT_DIR}/../../..)
echo $ROOT_DIR

MEGATRON_PATH=${ROOT_DIR}
export PYTHONPATH=${PYTHONPATH}:${ROOT_DIR}/Megatron-LM/${MCORE_VERSION:-core_r0_10_0_klx}:${ROOT_DIR}/megatron_patch/${MEGATRON_PATCH_VERSION:-megatron_patch_250328}

MODEL_SIZE=${MODEL_SIZE:-A2.4B}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-128}
LR=${LR:-1e-4}
MIN_LR=${MIN_LR:-1e-5}
SEQ_LEN=${SEQ_LEN:-4096}
PAD_LEN=${SEQ_LEN}
LR_DECAY_STYLE=cosine
WEIGHT_DECAY=0.1
EXTRA_VOCAB_SIZE=2400
if [ "${MODEL_SIZE}" = "A21B" ]; then
    MP_PP0_LAYERS=${MP_PP0_LAYERS:-4}
fi
TP=${TP:-1}
PP=${PP:-1}
EP=${EP:-8}
CP=${CP:-1}
AC=${AC:-none}
TASK=${TASK:-pretrain} # pretrain/sft
DATASET_TYPE=${DATASET_TYPE:-mmap}  # mmap /raw
SAVE_INTERVAL=${SAVE_INTERVAL:-5}
STORAGE_PATH=${STORAGE_PATH:-/mnt/v5000-megatron/v5000-megatron}
PRETRAIN_CHECKPOINT_PATH=${PRETRAIN_CHECKPOINT_PATH:-"${STORAGE_PATH}/model/DeepSeek-V2-Lite"}
DATASET_PATH=${DATASET_PATH:-"${STORAGE_PATH}/dataset/deepseek-v2/mmap_deepseekv2_datasets_text_document"}
VALID_DATASET_PATH=${DATASET_PATH}
if [[ -z ${OUTPUT_DIR} ]];then
    OUTPUT_BASEPATH=${ROOT_DIR}/output
else
    OUTPUT_BASEPATH=${OUTPUT_DIR}
fi
MP_SFT_PACKING=false
CPT_CONTINUE=${CPT_CONTINUE:-false}
ASYNC_SAVE=false
USE_VIRTUAL_PP=false
USE_SWA=false
USE_FP8=false
PR=${PR:-bf16}
DO=true
FL=true
SP=true
TE=true
OPTIMIZER_OFFLOAD=${OPTIMIZER_OFFLOAD:-false}
SAVE_CKPT=${SAVE_CKPT:-true}
# RMS_NORM_EPS=1e-6
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_GID_INDEX=3
if [ $ENV = dsw ]; then

    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
    GPUS_PER_NODE=`echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}'`
    # Change for multinode config
    MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
    NODE_RANK=${RANK:-0}
    if [ "$NODE_RANK" -eq 0 ] && [ $MASTER_ADDR = "localhost" ]; then
            MASTER_ADDR=${POD_NAME}
    fi
    echo "MASTER_ADDR is ${MASTER_ADDR}"
    NNODES=${WORLD_SIZE:-1}
    GPUS_PER_NODE=${TQ_GPU_NUM:-8}
    MASTER_PORT=${MASTER_PORT:-9988}

elif [ $ENV = dlc ]; then
NNODES=${WORLD_SIZE}
NODE_RANK=${RANK:-0}
GPUS_PER_NODE=${TQ_GPU_NUM}
fi

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"


if [ $USE_FP8 = true ]; then
    PR=fp8
fi


if echo "${DATASET_PATH}" | grep -q -E '\.txt$'
then
    DATASET_FILE=$DATASET_PATH
    DATASET_PATH="$(grep -v '^#' ${DATASET_FILE})"
    data_cache_options=" \
        --data-cache-path $OUTPUT_BASEPATH/data_cache"
else
    data_cache_options=" \
            "
fi
data_cache_options=" \
        --data-cache-path $OUTPUT_BASEPATH/deepseek-v2/data_cache"


if [ $DATASET_TYPE = mmap ]; then
    dataset_type_options=" \
		    --dataset MMAP \
            --data-path ${DATASET_PATH} \
            --split 99,1,0 "
elif [ $DATASET_TYPE = raw ]; then
    dataset_type_options=" \
        --train-data-path ${DATASET_PATH} \
        --valid-data-path ${VALID_DATASET_PATH} \
        --dataloader-type cyclic \
        --dataset LLama-SFT-Raw "
fi

if [ $MODEL_SIZE = A2.4B ]; then

HIDDEN_SIZE=2048
NUM_ATTN_HEADS=16
NUM_LAYERS=27
INTERMEDIATE_SIZE=10944
MOE_INTERMEDIATE_SIZE=1408
MAX_POSITION_EMBEDDINGS=${SEQ_LEN}
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
    --expert-model-parallel-size ${EP} \
    --kv-lora-rank ${KV_LORA_RANK} \
    --qk-head-dim ${QK_NOPE_HEAD_DIM} \
    --qk-pos-emb-head-dim ${QK_ROPE_HEAD_DIM} \
    --v-head-dim ${V_HEAD_DIM} \
    --moe-router-load-balancing-type aux_loss"

elif [ $MODEL_SIZE = A21B ]; then

HIDDEN_SIZE=5120
NUM_ATTN_HEADS=128
NUM_LAYERS=60
INTERMEDIATE_SIZE=12288
MOE_INTERMEDIATE_SIZE=1536
MAX_POSITION_EMBEDDINGS=${SEQ_LEN}
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
    --moe-ffn-hidden-size ${MOE_INTERMEDIATE_SIZE} \
    --moe-router-topk ${ROUTER_TOPK} \
    --num-experts ${NUM_EXPERTS} \
    --moe-layer-freq ${MOE_LAYER_FREQ} \
    --moe-aux-loss-coeff 1e-2 \
    --moe-shared-expert-intermediate-size $((${MOE_INTERMEDIATE_SIZE} * ${NUM_SHARED_EXPERTS} )) \
    --expert-model-parallel-size ${EP} \
    --q-lora-rank ${Q_LORA_RANK} \
    --kv-lora-rank ${KV_LORA_RANK} \
    --qk-head-dim ${QK_NOPE_HEAD_DIM} \
    --qk-pos-emb-head-dim ${QK_ROPE_HEAD_DIM} \
    --v-head-dim ${V_HEAD_DIM} \
    --moe-router-load-balancing-type aux_loss"

fi

if [ ${PP} -gt 1 ] && [ ${USE_VIRTUAL_PP} = true ]; then
    if [ $((NUM_LAYERS % PP)) -eq 0 ] && [ $((NUM_LAYERS / PP % 4)) -eq 0 ]; then
        VIRTUAL_PP=$((NUM_LAYERS / PP / 4))
        virtual_pp_options="--num-layers-per-virtual-pipeline-stage ${VIRTUAL_PP}"
    elif [ $((NUM_LAYERS % PP)) -eq 0 ] && [ $((NUM_LAYERS / PP % 2)) -eq 0 ]; then
        VIRTUAL_PP=$((NUM_LAYERS / PP / 2))
        virtual_pp_options="--num-layers-per-virtual-pipeline-stage ${VIRTUAL_PP}"
    else
        virtual_pp_options=""
    fi
else
    virtual_pp_options=""
fi

comm_overlap_option=""
# comm_overlap_option="\
#     --overlap-grad-reduce \
#     --overlap-param-gather"

if [ -z ${MP_AC_LAYERS} ];then
    MP_AC_LAYERS=1
fi

if [ $AC = full ]; then
    _check=$(( ($NUM_LAYERS / $PP) % ${MP_AC_LAYERS} ))
    if [ $_check != 0 ]; then
        echo "the num layers per pp rank must be a multiple of the recompute layers."
        exit -1
    fi
    activation_checkpoint_options=" \
                    --recompute-method ${RECOMPUTE_METHOD:-block} \
            --recompute-num-layers ${MP_AC_LAYERS} \
                    --recompute-granularity full"
elif [ $AC = sel ]; then
    activation_checkpoint_options=" \
        --recompute-activations"
elif [ $AC = none ]; then
    activation_checkpoint_options=" \
    "
elif [ $AC = offload ]; then
    activation_checkpoint_options=" \
                    --cpu-offloading \
                    --cpu-offloading-num-layers ${MP_AC_LAYERS}"
    if [ $TP_COMM_OVERLAP -eq 1 ]; then
        echo "Disable --overlap-grad-reduce and --overlap-param-gather when cpu offloading is on..."
        comm_overlap_option="\
            --tp-comm-overlap"
    else
        echo "Disable --overlap-grad-reduce and --overlap-param-gather when cpu offloading is on..."
        comm_overlap_option=""
    fi
fi

if [ $PR = fp16 ]; then
    pr_options=" \
                    --fp16 \
            --apply-query-key-layer-scaling"
    export NVTE_APPLY_QK_LAYER_SCALING=1
elif [ $PR = bf16 ]; then
    pr_options=" \
        --bf16"
elif [ $PR = fp8 ]; then
    pr_options=" \
        --bf16 \
        --fp8-format hybrid \
        --fp8-amax-compute-algo max \
        --fp8-amax-history-len 1024"
fi

if [ $OPTIMIZER_OFFLOAD != false ] && [ $DO = false ]; then
    echo "Offload optimizer is valid only if \$DO=true"
    DO=true
fi

if [ $DO = true ]; then
    do_options=" \
                    --use-distributed-optimizer"

elif [ $DO = false ]; then
    do_options=" \
                    "
fi

if [ $FL = true ]; then
    export NVTE_FLASH_ATTN=1 NVTE_FUSED_ATTN=0
    attn_backend_option=" \
       --use-flash-attn \
    "
elif [ $FL = false ]; then
    export NVTE_FLASH_ATTN=0 NVTE_FUSED_ATTN=1
    attn_backend_option=""
fi

if [ $TE = true ]; then
    te_options=" \
		    --transformer-impl transformer_engine"

elif [ $TE = false ]; then
    te_options=" \
        --transformer-impl local"
fi

if [ $SP = true ] && [ $TP -gt 1 ]; then
    sp_options=" \
                    --sequence-parallel"

elif [ $SP = false ]; then
    sp_options=" \
                    "
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
        --decoder-first-pipeline-num-layers ${MP_PP0_LAYERS}
    "
else
    echo "uneven pipeline split must be used when PP > 1"
    exit -1
fi


if [ -z ${MP_SFT_PACKING} ]; then
    MP_SFT_PACKING=false
fi

if [ ${MP_SFT_PACKING} = true ]; then
    packing_options=" \
        --reset-position-ids \
        --no-create-attention-mask-in-dataloader
    "
else
    packing_options=""
fi

if [ ${USE_SWA} = true ]; then
    WINDOW_SIZE=$((SEQ_LEN / 8))
    swa_options=" \
        --window-size ${WINDOW_SIZE} 0 \
    "
else
    swa_options=""
fi

if [ -z ${ASYNC_SAVE} ]; then
    ASYNC_SAVE=false
fi

if [ ${ASYNC_SAVE} = true ]; then
    async_save_options=" \
        --async-save \
        --use-dist-ckpt
    "
else
    async_save_options=""
fi


if [ $TASK = pretrain ]; then
    task_options=" \
            --train-mode pretrain "
elif [ $TASK = sft ]; then
    task_options=" \
        --train-mode finetune \
        --eod-mask-loss "
fi

TRAIN_TOKENS=12174728086
WARMUP_TOKENS=243494561
#TRAIN_ITERS=$(( ${TRAIN_TOKENS} / ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))
TRAIN_ITERS=${TRAIN_ITERS:-10}
LR_WARMUP_ITERS=$(( ${WARMUP_TOKENS}  / ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))
LR_DECAY_ITERS=$(( ${TRAIN_TOKENS} /  ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))

TASK_NAME="mcore-deepseek-v2-${MODEL_SIZE}-${TASK}"
CURRENT_TIME=$(date +"%m-%d-%H-%M")
DETAIL_TASK_NAME="${TASK_NAME}-lr-${LR}-minlr-${MIN_LR}-bs-${MICRO_BATCH_SIZE}-gbs-${GLOBAL_BATCH_SIZE}-seqlen-${SEQ_LEN}-pr-${PR}-tp-${TP}-pp-${PP}-cp-${CP}-virtual_pp-${VIRTUAL_PP}-ac-${AC}-do-${DO}-sp-${SP}-ep-${EP}/${CURRENT_TIME}"

SAVED_PRETRAIN_CHECKPOINT_PATH="${OUTPUT_BASEPATH}/checkpoint/${TASK_NAME}-TP${TP}-PP${PP}"
LOG_DIR=${OUTPUT_BASEPATH}/${DETAIL_TASK_NAME}
LOG_NAME="${NODE_RANK}.txt"
TENSORBOARD_DIR="${OUTPUT_BASEPATH}/${DETAIL_TASK_NAME}"


mkdir -p ${SAVED_PRETRAIN_CHECKPOINT_PATH}
mkdir -p ${LOG_DIR}
mkdir -p ${TENSORBOARD_DIR}


if [ $SAVE_CKPT = true ]; then
    save_ckpt_options=" \
        --save ${SAVED_PRETRAIN_CHECKPOINT_PATH} \
        --ckpt-format torch "
fi


if [ -z ${CPT_CONTINUE} ] || [ ${CPT_CONTINUE} = false ]; then
    cpt_continue_options="\
     --no-load-optim \
     --no-load-rng "
elif [ ${CPT_CONTINUE} = true ];  then
    #PRETRAIN_CHECKPOINT_PATH=${SAVED_PRETRAIN_CHECKPOINT_PATH}
    cpt_continue_options="\
        --no-load-rng "
fi

if [ -e  ${SAVED_PRETRAIN_CHECKPOINT_PATH}/latest_checkpointed_iteration.txt ]; then
    echo "${SAVED_PRETRAIN_CHECKPOINT_PATH}/latest_checkpointed_iteration.txt 文件存在"
    PRETRAIN_CHECKPOINT_PATH=${SAVED_PRETRAIN_CHECKPOINT_PATH}
else
    echo "${SAVED_PRETRAIN_CHECKPOINT_PATH} :文件夹为空"
    find -L ${PRETRAIN_CHECKPOINT_PATH} -maxdepth 1 -type f -name "*.json" -print0 | xargs -0 cp -v -f -t ${SAVED_PRETRAIN_CHECKPOINT_PATH}
fi

if [ $PRETRAIN_CHECKPOINT_PATH != none ]; then
    load_options=" \
            --load $PRETRAIN_CHECKPOINT_PATH \
            --auto-detect-ckpt-format"
fi

megatron_options="  \
        --lr ${LR} \
        --min-lr ${MIN_LR} \
        --lr-decay-style ${LR_DECAY_STYLE} \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --weight-decay ${WEIGHT_DECAY} \
        --clip-grad 1.0 \
        --init-method-std 0.008 \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --lr-decay-iters ${LR_DECAY_ITERS} \
        --lr-warmup-iters ${LR_WARMUP_ITERS} \
        --train-iters ${TRAIN_ITERS} \
        --micro-batch-size ${MICRO_BATCH_SIZE} \
        --global-batch-size ${GLOBAL_BATCH_SIZE} \
        --num-layers ${NUM_LAYERS} \
        --hidden-size ${HIDDEN_SIZE} \
        --num-attention-heads ${NUM_ATTN_HEADS} \
        --ffn-hidden-size ${INTERMEDIATE_SIZE} \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
        --max-padding-length ${PAD_LEN} \
        --log-interval 1 \
        --log-throughput \
        --eval-interval 10000 \
        --eval-iters 0 \
        --save-interval ${SAVE_INTERVAL} \
        --tensorboard-queue-size 1 \
        --tensorboard-dir ${TENSORBOARD_DIR} \
        --log-timers-to-tensorboard \
        --log-validation-ppl-to-tensorboard \
        --tensor-model-parallel-size ${TP} \
        --pipeline-model-parallel-size ${PP} \
        --context-parallel-size ${CP} \
        --num-workers 8 \
        --extra-vocab-size ${EXTRA_VOCAB_SIZE} \
        --patch-tokenizer-type DeepSeekV2Tokenizer \
        --swiglu \
        --normalization RMSNorm \
        --norm-epsilon ${RMS_NORM_EPS} \
        --use-rotary-position-embeddings \
        --no-bias-swiglu-fusion \
        --no-rope-fusion \
        --position-embedding-type rope \
        --untie-embeddings-and-output-weights \
        --disable-bias-linear \
        --rotary-base ${ROPE_THETA} \
        --rotary-scaling-factor ${SCALE_FACTOR} \
        --kv-channels ${V_HEAD_DIM} \
        --qk-layernorm \
        --multi-latent-attention \
        --moe-token-dispatcher-type alltoall \
        --sequence-parallel \
        --use-cpu-initialization \
        --log-mfu \
        --mfu-base-value 312 \
        --log-memory-to-tensorboard \
        --log-token-throughput \
        "

if [[ -z ${LOG_FILE} ]];then
  LOG_FILE=${LOG_DIR}/${LOG_NAME}
fi      
if [ "${MODEL_SIZE}" = "A21B" ]; then
    megatron_options="${megatron_options} \
        --use-distributed-optimizer \
    "
fi  

run_cmd="env;torchrun $DISTRIBUTED_ARGS ./V5000/deepseek_v2/pretrain_deepseek.py
 ${megatron_options} \
 ${pr_options} \
 ${load_options} \
 ${te_options} \
 ${activation_checkpoint_options} \
 ${do_options} \
 ${attn_backend_option} \
 ${async_save_options} \
 ${sp_options} \
 ${moe_options} \
 ${dataset_type_options} \
 ${offload_option} \
 ${comm_overlap_option} \
 ${task_options} \
 ${packing_options} \
 ${cpt_continue_options} \
 ${data_cache_options} \
 ${virtual_pp_options} \
 ${swa_options} \
 ${save_ckpt_options} \
 ${uneven_split_option} \
 2>&1 | tee ${LOG_FILE}
 "
echo ${run_cmd}
eval ${run_cmd} 
status=$?
[ $status -ne 0 ] && dmesg -T >> ${LOG_DIR}/dmesg_${RANK}.log
exit $status
