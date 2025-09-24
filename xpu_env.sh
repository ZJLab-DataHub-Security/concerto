# XPU相关参数
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=mlx5
export NCCL_IB_GID_INDEX=3
export CUDA_DEVICE_ORDER=OAM_ID
#################################
export CUDART_DUMMY_REGISTER=1
export XPU_FORCE_USERMODE_LAUNCH=1
export XMLIR_DIST_SINGLETON_STREAM=true
export DIST_MULTI_STREAM=${DIST_MULTI_STREAM:-true}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-8}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}
export XMLIR_FA_GEMM_TYPE=float
export XBLAS_FC_HBM_VERSION=40
export XMLIR_PARALLEL_SAVE_MEMORY=${XMLIR_PARALLEL_SAVE_MEMORY:-false}
export XMLIR_DISABLE_CUDA_ALLOCATOR=true

#################################
export XMLIR_XDNN_PYTORCH_CHECK_ENABLE_FALLBACK_BOOL=0
export XMLIR_ENABLE_FALLBACK_TO_CPU_BOOL=False
export XMLIR_DUMP_FALLBACK_OP_LIST_BOOL=true
export XMLIR_DIST_ASYNC_ISEND_IRECV=true
##################
# bf16类型专用(megatron相关变量参考<百舸megatron专用>)
##################
export USE_FAST_BF16_FC=true # 仅bf16下用到
# export USE_CAST_FC_FUSION=true # 仅bf16下用到, fp16转bf16与fc计算融合算子
export XMLIR_BATCH_PARALLEL=${XMLIR_BATCH_PARALLEL:-true}
export FC_DW_MULTI_STREAM=${FC_DW_MULTI_STREAM:-true}
export XPU_FORCE_SHARED_DEVICE_CONTEXT=1
#################
# 通信通用
##################
##################
export BKCL_RDMA_PROXY_DISABLE=1
export BKCL_USE_AR=1
export BKCL_RING_OPT=1
export BKCL_FLAT_RING=1
export BKCL_CCIX_RING=1
export BKCL_TREE_THRESHOLD=1
export BKCL_CCIX_BUFFER_GM=1
export BKCL_FORCE_L3_RDMA=0
export BKCL_RING_BUFFER_GM=1
export BKCL_ENABLE_XDR=1
export BKCL_RDMA_FORCE_TREE=1
export BKCL_TREE_THRESHOLD=1
export BKCL_XLINK_D2D=0
export BKCL_XLINK_ETH=0
export BKCL_XLINK_C2C=1
export BKCL_TRANS_UNSUPPORTED_DATATYPE=1
export BKCL_KL3_TURBO_MODE=1
export BKCL_RING_BUFFER_SIZE=2097152
export ALLREDUCE_FUSION=0
unset BKCL_KL3_SYSCON_FLAG

# export BKCL_SOCKET_IFNAME=bond0
export BKCL_RDMA_NICS=bond2,bond2,bond3,bond3,bond4,bond4,bond5,bond5
export BKCL_TIMEOUT=${BKCL_TIMEOUT:-1800}
export CUDA_DISABLE_PRINTF=${CUDA_DISABLE_PRINTF:-1}
export BKCL_RDMA_VERBS=${BKCL_RDMA_VERBS:-1}

export XME_USE_LOCAL_TE=true
export XME_USE_TE_VERSION=1.7.0
export XME_USE_CUSTOM_TRAINING_LOG=true
export XME_FORCE_SYNC_D2H_COPY=true
export ENABLE_GROUPED_GEMM=${ENABLE_GROUPED_GEMM:-true}
export SAVE_INTERVAL=${SAVE_INTERVAL:-50}


if dmesg -T |  grep -q "noc_idle" <(cat /dev/stdin); then
    echo "Error: noc timeout found in dmesg, please check node ${RANK} ${HOSTNAME} ${POD_IP}!"
    dmesg -T >> dmesg_${RANK}.log
    exit 999
fi

# 设置任务超时检测阈值为30分钟,默认20分钟
for dev_id in $(echo $CUDA_VISIBLE_DEVICES | tr ',' ' '); do
    if [ -f /proc/kunlun/dev$dev_id/task_timeout_detect_threshold_in_ms ]; then
        echo 1800000 > /proc/kunlun/dev$dev_id/task_timeout_detect_threshold_in_ms
        cat /proc/kunlun/dev$dev_id/task_timeout_detect_threshold_in_ms
    fi
done