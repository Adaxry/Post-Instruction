
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_NET_GDR_READ=1
export CXX=g++

export MASTER_ADDR="${CHIEF_IP:=localhost}"
#export MASTER_PORT="${MASTER_PORT:=29500}"
MASTER_PORT=$((1 + $RANDOM % 99999))

gpu_num=4
if [ $gpu_num -eq 4 ]; then
    accum_num=32
elif [ $gpu_num -eq 8 ]; then
    accum_num=16
fi
model_name=$your_model_name
work_dir=$your_work_dir
train_path=./run_clm_llms.py
premodel=${work_dir}/bloomz-7b1-mt
model_save=$work_dir/checkpoint/$model_name
LOG_FILE=${work_dir}/finetune/log.${model_name}

data_dir=$your_data_dir
export TRANSFORMERS_CACHE=${data_dir}/cache/
export HF_HOME=${data_dir}/cache/
export TORCH_EXTENSIONS_DIR=${data_dir}/cache/torch_extension/${model_name}
export OMP_NUM_THREADS=20
TOKENIZERS_PARALLELISM=false
# HOST_NUM will be 1
HOST_NUM=1
INDEX=0

train_files=../data/data_alpaca_gpt4_hf_en_post_ins.json,../data/parrot_hint_data_hf_post_ins.json,../data/wmt/post_ins_formetted/newtest16_20.de2en.fix_cxt3.post_ins.hf.json,../data/wmt/post_ins_formetted/newtest17_20.cn2en.fix_cxt3.post_ins.hf.json,../data/wmt/post_ins_formetted/newtest17_20.en2cn.fix_cxt3.post_ins.hf.json,../data/wmt/post_ins_formetted/newtest17_20.en2de.fix_cxt3.post_ins.hf.json
 

torchrun --nnodes $HOST_NUM --node_rank $INDEX --nproc_per_node $gpu_num \
    --master_addr $MASTER_ADDR --master_port $MASTER_PORT  \
    ${train_path} \
    --deepspeed ./deepspeed_config.json \
    --model_name_or_path ${premodel} \
    --train_file $train_files \
    --preprocessing_num_workers 16 \
    --dataloader_num_workers 8 \
    --dataloader_pin_memory True \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $accum_num \
    --num_train_epochs 1. \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 50 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --block_size 768 \
    --do_train \
    --evaluation_strategy "no" \
    --validation_split_percentage 0 \
    --fp16 True \
    --fp16_full_eval True \
    --streaming \
    --ddp_timeout 3600 \
    --seed 1 \
    --gradient_checkpointing False \
    --output_dir ${model_save} \
    --cache_dir ${data_dir}/cache/ \
    --freeze_emb True \
    --overwrite_output_dir \
    --overwrite_cache \
    2>&1 |tee ${LOG_FILE}

    #--gradient_checkpointing True \
