# for machine translation
python convert_pair_to_alpaca.py \
	-sf $your_source_file \
	-tf $your_target_file \
	-s  $source_language -t $target_language \
	-if instruct_follow.filtered.trg_zh.post_ins.txt \
	-of $output_name.json  \
	--fix-cxt 3
python convert_alpaca_to_hf.py \
	-i $output_name.json \
	-o $output_name.hf.json \
	--above-prompt  # for post-ins


# for text summarization
python convert_pair_to_alpaca.py \
	-sf $your_source_file \
	-tf $your_target_file \
	-s  en -t en \
	-if instruct_follow_summary_post_ins.txt \
	-of $output_name.json  \
	--fix-cxt 3

python convert_alpaca_to_hf.py \
	-i $output_name.json \
	-o $output_name.hf.json \
	--above-prompt  # for post-ins
