CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
bash ${CURRENT_DIR}/run_8xH20.sh A3B /mnt/zj-gpfs/model/qwen/Qwen3-30B-A3B /mnt/zj-gpfs/home/qianhao/models/megatron_ckpt/mcore_qwen3_a3b_t4_e8_test false true bf16 /mnt/zj-gpfs/model/qwen/Qwen3-30B-A3B

# examples for convert a3b with shared_experts
#bash ${CURRENT_DIR}/run_8xH20.sh A3BS /mnt/geogpt-training/home/shuangfeng/models/Qwen3-30B-A3B-Instruct-2507-shared8-gpavg/ /mnt/geogpt-training/home/qianhao/models/megatron_ckpt/mcore_qwen3_a3b_t4_p2_e4_add_shared_experts false true bf16 /mnt/geogpt-training/home/shuangfeng/models/Qwen3-30B-A3B-Instruct-2507-shared8-gpavg/
