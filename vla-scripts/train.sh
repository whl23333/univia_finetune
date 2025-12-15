export LD_LIBRARY_PATH=/home/pai/envs/openvla/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
GPUS_PER_NODE=1  
NNODES=1
MASTER_PORT=${MASTER_PORT:-28596}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
RANK=${RANK:-0}


# Run your training script with torchrun
torchrun --nproc_per_node ${GPUS_PER_NODE} --nnodes ${NNODES} --node_rank ${RANK} --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} train.py \
                                 --vla.type prism-dinosiglip-224px+ft-calvin \
                                 --run_root_dir "vla_calvin_single_view_log" \
                                 --data_root_dir "/data/share/calvin/task_ABC_D" \
                                 --pretrain_vlm /data/250010208/whl/UniVLA/checkpoints/pretrained_unvla/