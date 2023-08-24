#!/bin/bash

work_dir=$your_work_dir
model_name=$your_model_name

post_ins_test=""
no_error_test=""
post_ins_test="_post_ins_test"
no_error_test="_no_error"  
#double_ins_test="_double_ins_test"  
no_repeat_ngram_size=12

start_step=200
sep_step=200
end_step=2400

result_dir=$work_dir/results/${model_name}${post_ins_test}${no_error_test}
log_name=log

if [ $no_repeat_ngram_size -gt 0 ]; then
  result_dir="${result_dir}_ngram${no_repeat_ngram_size}"
fi

ins_file="instruct_inf"
template="prompt_input"
post_ins_file=""

if [ -n "$no_error_test" ]; then
  ins_file="instruct_inf_e2t"
else
  ins_file="instruct_inf"
fi

if [ -n "$post_ins_test" ]; then
  template="prompt_input_above"
  ins_file=${ins_file}_above 
fi

if [ -n "$double_ins_test" ]; then
  template="double_prompt_input"
  post_ins_file=${ins_file}_above 
fi

ins_file=${ins_file}.txt 
post_ins_file=${post_ins_file}.txt 

echo "Debug: use ins_file = ", $ins_file
echo "Debug: use post_ins_file = ", $post_ins_file
echo "Debug: use template = ", $template


if [ ! -d $result_dir ]; then
    mkdir -p $result_dir
    chmod 777 $result_dir -R
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TRANSFORMERS_CACHE=/apdcephfs/share_47076/yijinliu/transformers/data/cache
export HF_HOME=/apdcephfs/share_47076/yijinliu/transformers/data/cache
export TF_ENABLE_ONEDNN_OPTS=0
python=/opt/anaconda3/bin/python3 

inference()
{
    src=$1
    tgt=$2
    step=$3
    echo "src" $src
    echo "tgt" $tgt
    echo "step" $step
    log_name=${log_name}_step${step}
    ${python} $work_dir/test/inference.py \
        --model-name-or-path ${work_dir}/checkpoint/$model_name/checkpoint-${step} \
        -lp $src-$tgt \
        -t 0.1 \
        -b 4 \
        -sa 'beam' \
        --batch 1 \
        --no-repeat-ngram-size $no_repeat_ngram_size \
        -ins $work_dir/test/$ins_file \
        -i $work_dir/test/WMT22/newstest22.$src-$tgt.$src \
        -tp $template \
        -o $result_dir/${src}-${tgt}_step${step}.out 

    ${python} sacre_verbose.py $result_dir/${src}-${tgt}_step${step}.out.hyp $work_dir/test/WMT22/newstest22.${src}-${tgt}.${tgt} $result_dir/tmp_bleu $tgt >> $result_dir/$log_name
}


for step in `seq $start_step $sep_step $end_step`; do

echo "step=" $step

(export CUDA_VISIBLE_DEVICES=0;inference de en $step;sleep 150)& \
(export CUDA_VISIBLE_DEVICES=1;inference en de $step)& \
(export CUDA_VISIBLE_DEVICES=2;inference en zh $step;sleep 150)& \
(export CUDA_VISIBLE_DEVICES=3;inference zh en $step)
wait
done
