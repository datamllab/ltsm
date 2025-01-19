data_name="
        web_attack"
save_format="pth.tar"


python ./ltsm/prompt_reader/stat_prompt/prompt_generate_split.py \
        --dataset_name ${data_name} \
        --save_format ${save_format}
python ./ltsm/prompt_reader/stat_prompt/prompt_normalization_split.py --mode fit --dataset_name ${data_name} --save_format ${save_format}
python ./ltsm/prompt_reader/stat_prompt/prompt_normalization_split.py --mode transform --dataset_name ${data_name} --save_format ${save_format}