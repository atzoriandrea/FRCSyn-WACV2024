export OMP_NUM_THREADS=32
CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 \
--node_rank=0 --master_addr="127.0.0.1" --master_port=1236 train.py
ps -ef | grep "train" | grep -v grep | awk '{print "kill -9 "$2}' | sh
