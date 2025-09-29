#!/bin/bash
set -eo pipefail 

ENV=dsw
ROOT_DIR=$( dirname -- "$( readlink -f -- "$0"; )"; )
ROOT_DIR=${ROOT_DIR}/../.. #v5000_megatron
echo $ROOT_DIR

MEGATRON_PATH=/workspace/Megatron-LM/
#export LD_LIBRARY_PATH=/mnt/v5000-megatron/v5000-megatron/liusong/output/so:$LD_LIBRARY_PATH
export PYTHONPATH=${PYTHONPATH}:${ROOT_DIR}:${MEGATRON_PATH}

### BASE CONFIG ###
MODEL_SIZE=${MODEL_SIZE:-A3B}
BATCH_SIZE=1
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-32}
LR=${LR:-1e-5}
MIN_LR=${MIN_LR:-1e-6}
SEQ_LEN=${SEQ_LEN:-32768}
PAD_LEN=${SEQ_LEN}
PR=${PR:-bf16}
LOAD_OPTIM=${LOAD_OPTIM:-true}
LOAD_RNG=${LOAD_RNG:-true}
MOE_AUX_LOSS_COEFF=${MOE_AUX_LOSS_COEFF:-0.001}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
### BASE CONFIG ###

### PARALLEL / BOOL OPTION ###
TP=${TP:-4}
PP=${PP:-1}
EP=${EP:-8}
ETP=${ETP:-1}
CP=1
SP=true
DO=true
FL=true
SFT=true
# MP_PP0_LAYERS=0
### PARALLEL / BOOL OPTION ###

### OTHERS ###
AC=${AC:-full}
ONLINE_PACKING=${ONLINE_PACKING:-false}
CHANNEL_LOSS=${CHANNEL_LOSS:-false}
FREEZE_MOE_ROUTER=${FREEZE_MOE_ROUTER:-false}
RECOMPUTE_METHOD=${RECOMPUTE_METHOD:-block}
MP_AC_LAYERS=${MP_AC_LAYERS:-46}
OPTIMIZER_OFFLOAD=${OPTIMIZER_OFFLOAD:-false}
LR_WARMUP=${LR_WARMUP:-0.1}
SAVE_INTERVAL=${SAVE_INTERVAL:-100}
PRETRAIN_CHECKPOINT_PATH=${PRETRAIN_CHECKPOINT_PATH:-/mnt/zj-gpfs/home/qianhao/models/megatron_ckpt/mcore_qwen3_a3b_t4_e8}
#DATASET_PATH=${DATASET_PATH:-/mnt/zj-gpfs/home/qianhao/data/tianqing-sample/sft-sample.jsonl}
DATASET_PATH=${DATASET_PATH:-/mnt/zj-gpfs/home/qianhao/data/qwen3_32b_sample_long_sft_32k_pack_text_document}
VALID_DATASET_PATH=${DATASET_PATH}

MP_SFT_PACKING=true
# CPT_CONTINUE=false

# the following two values will not be used when SFT is true
TRAIN_SAMPLES=${TRAIN_SAMPLES:-2000} 
###############################

if [[ -z ${OUTPUT_DIR} ]];then
    OUTPUT_BASEPATH=${ROOT_DIR}/output
else
    OUTPUT_BASEPATH=${OUTPUT_DIR}
fi
SAVE_CKPT=true

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_GID_INDEX=3
if [ $ENV = dsw ]; then

    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}
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

if [ $FL = true ]; then
    export NVTE_FLASH_ATTN=1 NVTE_FUSED_ATTN=0
    attn_backend_option=" \
        --attention-backend flash \
    "
elif [ $FL = false ]; then
    export NVTE_FLASH_ATTN=0 NVTE_FUSED_ATTN=1
    attn_backend_option=" \
        --attention-backend fused \
    "
fi

if [ $MODEL_SIZE = 0.6B ]; then
    NUM_LAYERS=28
    HIDDEN_SIZE=1024
    NUM_ATTENTION_HEADS=16
    INTERMEDIATE_SIZE=3072
    NUM_KEY_VALUE_HEADS=8
    MAX_POSITION_EMBEDDINGS=40960
    EXTRA_VOCAB_SIZE=293
    ROPE_THETA=1000000
    RMS_NORM_EPS=1e-6
    gqa_options=" \
                --group-query-attention \
                --num-query-groups ${NUM_KEY_VALUE_HEADS}"
    
    tie_option=""
    moe_options=""
elif [ $MODEL_SIZE = 1.7B ]; then
    NUM_LAYERS=28
    HIDDEN_SIZE=2048
    NUM_ATTENTION_HEADS=16
    INTERMEDIATE_SIZE=6144
    NUM_KEY_VALUE_HEADS=8
    MAX_POSITION_EMBEDDINGS=40960
    EXTRA_VOCAB_SIZE=293
    ROPE_THETA=1000000
    RMS_NORM_EPS=1e-6
    gqa_options=" \
                --group-query-attention \
                --num-query-groups ${NUM_KEY_VALUE_HEADS}"
    
    tie_option=""
    moe_options=""
elif [ $MODEL_SIZE = 4B ]; then
    NUM_LAYERS=36
    HIDDEN_SIZE=2560
    NUM_ATTENTION_HEADS=32
    INTERMEDIATE_SIZE=9728
    NUM_KEY_VALUE_HEADS=8
    MAX_POSITION_EMBEDDINGS=40960
    EXTRA_VOCAB_SIZE=293
    ROPE_THETA=1000000
    RMS_NORM_EPS=1e-6
    gqa_options=" \
                --group-query-attention \
                --num-query-groups ${NUM_KEY_VALUE_HEADS}"
    
    tie_option=""
    moe_options=""
elif [ $MODEL_SIZE = 8B ]; then
    NUM_LAYERS=36
    HIDDEN_SIZE=4096
    NUM_ATTENTION_HEADS=32
    INTERMEDIATE_SIZE=12288
    NUM_KEY_VALUE_HEADS=8
    MAX_POSITION_EMBEDDINGS=40960
    EXTRA_VOCAB_SIZE=293
    ROPE_THETA=1000000
    RMS_NORM_EPS=1e-6
    gqa_options=" \
                --group-query-attention \
                --num-query-groups ${NUM_KEY_VALUE_HEADS}"
    
    tie_option=" \
            --untie-embeddings-and-output-weights \
            "
    moe_options=""
elif [ $MODEL_SIZE = 14B ]; then 
    NUM_LAYERS=40
    HIDDEN_SIZE=5120
    NUM_ATTENTION_HEADS=40
    INTERMEDIATE_SIZE=17408
    NUM_KEY_VALUE_HEADS=8
    MAX_POSITION_EMBEDDINGS=40960
    EXTRA_VOCAB_SIZE=293
    ROPE_THETA=1000000
    RMS_NORM_EPS=1e-6
    gqa_options=" \
                --group-query-attention \
                --num-query-groups ${NUM_KEY_VALUE_HEADS}"
    
    tie_option=" \
            --untie-embeddings-and-output-weights \
            "

    moe_options=""
elif [ $MODEL_SIZE = 32B ]; then
    NUM_LAYERS=64
    HIDDEN_SIZE=5120
    NUM_ATTENTION_HEADS=64
    INTERMEDIATE_SIZE=25600
    NUM_KEY_VALUE_HEADS=8
    MAX_POSITION_EMBEDDINGS=40960
    EXTRA_VOCAB_SIZE=293
    ROPE_THETA=1000000
    RMS_NORM_EPS=1e-6
    gqa_options=" \
                --group-query-attention \
                --num-query-groups ${NUM_KEY_VALUE_HEADS}"
    
    tie_option=" \
            --untie-embeddings-and-output-weights \
            "

    moe_options=""
elif [ $MODEL_SIZE = A3B ]; then
    HIDDEN_SIZE=2048
    NUM_ATTENTION_HEADS=32
    NUM_LAYERS=48
    INTERMEDIATE_SIZE=6144
    MOE_INTERMEDIATE_SIZE=768
    MAX_POSITION_EMBEDDINGS=40960
    EXTRA_VOCAB_SIZE=293
    NUM_KEY_VALUE_HEADS=4
    ROPE_THETA=1000000
    NUM_EXPERTS=128
    ROUTER_TOPK=8
    RMS_NORM_EPS=1e-6

    moe_options=" \
    	--moe-grouped-gemm \
        --moe-token-dispatcher-type allgather \
        --moe-router-topk ${ROUTER_TOPK} \
        --num-experts ${NUM_EXPERTS} \
        --expert-tensor-parallel-size ${ETP} \
        --expert-model-parallel-size ${EP} \
        --moe-ffn-hidden-size ${MOE_INTERMEDIATE_SIZE} \
        --moe-router-load-balancing-type aux_loss \
        --moe-aux-loss-coeff ${MOE_AUX_LOSS_COEFF} \
        --moe-layer-freq '([1]*48)' \
        "

    tie_option=" \
            --untie-embeddings-and-output-weights \
            "

    gqa_options=" \
                --group-query-attention \
                --num-query-groups ${NUM_KEY_VALUE_HEADS}"
elif [ $MODEL_SIZE = A22B ]; then
    HIDDEN_SIZE=4096
    NUM_ATTENTION_HEADS=64
    NUM_LAYERS=94
    INTERMEDIATE_SIZE=12288
    MOE_INTERMEDIATE_SIZE=1536
    MAX_POSITION_EMBEDDINGS=40960
    EXTRA_VOCAB_SIZE=293
    NUM_KEY_VALUE_HEADS=4
    ROPE_THETA=1000000
    NUM_EXPERTS=128
    ROUTER_TOPK=8
    RMS_NORM_EPS=1e-6

    moe_options=" \
    	--moe-grouped-gemm \
        --moe-token-dispatcher-type allgather \
        --moe-router-topk ${ROUTER_TOPK} \
        --num-experts ${NUM_EXPERTS} \
        --expert-tensor-parallel-size ${ETP} \
        --expert-model-parallel-size ${EP} \
        --moe-ffn-hidden-size ${MOE_INTERMEDIATE_SIZE} \
        --moe-router-load-balancing-type aux_loss \
        --moe-aux-loss-coeff ${MOE_AUX_LOSS_COEFF} \
        --moe-layer-freq '([1]*94)' \
        --moe-router-pre-softmax
        "

    tie_option=" \
            --untie-embeddings-and-output-weights \
            "

    gqa_options=" \
                --group-query-attention \
                --num-query-groups ${NUM_KEY_VALUE_HEADS}"
fi


# Here are some configs controled by env
if [ -z ${MP_DATASET_TYPE} ];then
    MP_DATASET_TYPE="idxmap"
fi

if [ -z ${MP_AC_LAYERS} ];then
    MP_AC_LAYERS=1
fi

if [ -z ${MP_VP} ]; then
    vp_option=""
else
    vp_option=" \
        --num-layers-per-virtual-pipeline-stage ${MP_VP}"
fi

if [ -z ${MP_SFT_PACKING} ]; then
    MP_SFT_PACKING=false
fi

TP_COMM_OVERLAP=$(( ($TP > 1) ? 1 : 0 ))
#comm_overlap_option="\
#    --overlap-grad-reduce \
#    --overlap-param-gather"

comm_overlap_option=""
#if [ $TP_COMM_OVERLAP -eq 1 ]; then
#    comm_overlap_option="\
#        --tp-comm-overlap \
#        --overlap-grad-reduce \
#        --overlap-param-gather"
#fi

if [ $AC = full ]; then
    _check=$(( ($NUM_LAYERS / $PP) % ${MP_AC_LAYERS} ))
    if [ $_check != 0 ]; then
        echo "the num layers per pp rank must be a multiple of the recompute layers."
    fi
    activation_checkpoint_options=" \
		    --recompute-method ${RECOMPUTE_METHOD} \
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
    do_option=" \
		    --use-distributed-optimizer"

elif [ $DO = false ]; then
    do_option=" \
                    "
fi


if [ $SP = true ] && [ $TP -gt 1 ]; then
    sp_option=" \
		    --sequence-parallel"

elif [ $SP = false ]; then
    sp_option=" \
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
        --decoder-first-pipeline-num-layers ${MP_PP0_LAYERS}"
else
    echo "uneven pipeline split must be used when PP > 1"
    exit -1
fi

if [ $PRETRAIN_CHECKPOINT_PATH != none ]; then
    load_option=" \
            --load $PRETRAIN_CHECKPOINT_PATH \
	    --auto-detect-ckpt-format"
fi

if [ $OPTIMIZER_OFFLOAD != false ]; then
    offload_option=" \
        --optimizer-cpu-offload \
        --use-precision-aware-optimizer \
        --optimizer-offload-fraction ${OPTIMIZER_OFFLOAD}"
fi

if [ $SFT = true ]; then
    TASK="sft"
    sft_options=" \
         --eod-mask-loss \
         --calculate-per-token-loss \
         --train-mode finetune"
else
    TASK="pretrain"
    sft_options=" \
        --train-mode pretrain"
fi

if [ ${MP_DATASET_TYPE} = "raw" ]; then
    dataset_options=" \
        --train-data-path ${DATASET_PATH} \
        --valid-data-path ${VALID_DATASET_PATH} \
        --dataloader-type cyclic \
        --dataset JSON-SFT "
else 
    dataset_options=" \
        --data-path ${DATASET_PATH} \
        --dataset MMAP \
        --dataloader-type cyclic \
        --split 99,1,0 "
fi
if [ ${ONLINE_PACKING} = true ]; then
    packing_options=" \
      --online-packing \
      --reset-position-ids \
      --no-create-attention-mask-in-dataloader "
    if [ ${CHANNEL_LOSS} = true ]; then
        packing_options=" \
          --online-packing \
          --reset-position_ids \
	      --no-create-attention-mask-in-dataloader \
	      --calc-channel-loss"
    fi
elif [ ${MP_SFT_PACKING} = true ]; then
    packing_options=" \
      --reset-position-ids \
      --no-create-attention-mask-in-dataloader "
else
    packing_options=" \
      --no-create-attention-mask-in-dataloader "
fi
if [ ${FREEZE_MOE_ROUTER} = true ]; then
    moe_options=" \
    ${moe_options} \
    --freeze-moe-router"
fi
##### Prepare logdirs #######
CURRENT_TIME=$(date +"%m-%d-%H:%M")
TASK_NAME="mcore-qwen3-${MODEL_SIZE}-${TASK}"
DETAIL_TASK_NAME="${TASK_NAME}-lr-${LR}-minlr-${MIN_LR}-bs-${BATCH_SIZE}-gbs-${GLOBAL_BATCH_SIZE}-seqlen-${SEQ_LEN}-pr-${PR}-tp-${TP}-pp-${PP}-cp-${CP}-ac-${AC}-do-${DO}-sp-${SP}/${CURRENT_TIME}${LABEL}"
TENSORBOARD_DIR="${OUTPUT_BASEPATH}/${DETAIL_TASK_NAME}"
SAVED_PRETRAIN_CHECKPOINT_PATH="${OUTPUT_BASEPATH}/checkpoint/${TASK_NAME}"
LOG_DIR=${OUTPUT_BASEPATH}/${DETAIL_TASK_NAME}
LOG_NAME="${NODE_RANK}.txt"

mkdir -p ${SAVED_PRETRAIN_CHECKPOINT_PATH}
mkdir -p ${LOG_DIR}
mkdir -p ${TENSORBOARD_DIR}

if [ $SAVE_CKPT = true ]; then
    save_ckpt_options=" \
        --save ${SAVED_PRETRAIN_CHECKPOINT_PATH} \
        --save-interval ${SAVE_INTERVAL} \
        --ckpt-format torch_dist "
fi

cpt_continue_options=""
if [ ${LOAD_OPTIM} = false ]; then
    cpt_continue_options=" \
	    ${cpt_continue_options} \
	    --no-load-optim "
fi
if [ ${LOAD_RNG} = false ]; then
    cpt_continue_options=" \
	    ${cpt_continue_options} \
	    --no-load-rng "
fi

find -L ${PRETRAIN_CHECKPOINT_PATH} -maxdepth 1 -type f -name "*.json" -print0 | xargs -0 cp -t ${SAVED_PRETRAIN_CHECKPOINT_PATH}


megatron_options="  \
        --lr ${LR} \
        --min-lr ${MIN_LR} \
        --lr-decay-style cosine \
        --weight-decay ${WEIGHT_DECAY} \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --clip-grad 1.0 \
        --init-method-std 0.008 \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --lr-warmup-fraction ${LR_WARMUP} \
        --train-samples ${TRAIN_SAMPLES} \
        --micro-batch-size ${BATCH_SIZE} \
        --global-batch-size ${GLOBAL_BATCH_SIZE} \
        --num-layers ${NUM_LAYERS} \
        --hidden-size ${HIDDEN_SIZE} \
        --num-attention-heads ${NUM_ATTENTION_HEADS} \
        --ffn-hidden-size ${INTERMEDIATE_SIZE} \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
        --max-padding-length ${PAD_LEN} \
        --log-interval 1 \
        --log-throughput \
	--log-mfu \
        --eval-interval 10000 \
        --eval-iters 0 \
        --tensorboard-queue-size 1 \
        --tensorboard-dir ${TENSORBOARD_DIR} \
        --log-timers-to-tensorboard \
        --log-validation-ppl-to-tensorboard \
        --tensor-model-parallel-size ${TP} \
        --pipeline-model-parallel-size ${PP} \
        --context-parallel-size ${CP} \
        --num-workers 8 \
        --extra-vocab-size ${EXTRA_VOCAB_SIZE} \
        --patch-tokenizer-type Qwen3Tokenizer \
        --swiglu \
        --normalization RMSNorm \
        --norm-epsilon ${RMS_NORM_EPS} \
        --use-rotary-position-embeddings \
        --position-embedding-type rope \
        --disable-bias-linear \
        --rotary-base ${ROPE_THETA} \
        --no-rope-fusion \
        --moe-token-dispatcher-type alltoall \
        --ckpt-format torch_dist \
        --transformer-impl transformer_engine \
        --cross-entropy-loss-fusion \
        --qk-layernorm \
        --kv-channels 128 \
        --use-cpu-initialization \
        "

if [[ -z ${LOG_FILE} ]];then
  LOG_FILE=${LOG_DIR}/${LOG_NAME}
fi

run_cmd="torchrun $DISTRIBUTED_ARGS run_qwen.py
 ${megatron_options} \
 ${dataset_options} \
 ${pr_options} \
 ${load_option} \
 ${activation_checkpoint_options} \
 ${do_option} \
 ${sp_option} \
 ${moe_options} \
 ${offload_option} \
 ${sft_options} \
 ${vp_option} \
 ${packing_options} \
 ${uneven_split_option} \
 ${attn_backend_option} \
 ${cpt_continue_options} \
 ${save_ckpt_options} \
 ${gqa_options} \
 ${tie_option} \
 2>&1 | tee ${LOG_FILE}
 "

echo ${run_cmd}
eval ${run_cmd}
set +x
