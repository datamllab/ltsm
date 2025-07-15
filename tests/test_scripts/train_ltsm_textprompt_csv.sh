TRAIN="datasets/ETT-small/ETTh1.csv 
    datasets/ETT-small/ETTh2.csv
    datasets/ETT-small/ETTm1.csv
    datasets/ETT-small/ETTm2.csv
    datasets/electricity/electricity.csv
    datasets/exchange_rate/exchange_rate.csv
    datasets/traffic/traffic.csv
    datasets/weather/weather.csv"

TEST="datasets/ETT-small/ETTh1.csv 
    datasets/ETT-small/ETTh2.csv
    datasets/ETT-small/ETTm1.csv
    datasets/ETT-small/ETTm2.csv
    datasets/electricity/electricity.csv
    datasets/exchange_rate/exchange_rate.csv
    datasets/traffic/traffic.csv
    datasets/weather/weather.csv"

PROMPT="prompt_bank/text_prompt_data_csv/csv_prompt.json"
epoch=1000
downsample_rate=20
freeze=0
lr=1e-3


for pred_len in 96 192 336 720
do
    OUTPUT_PATH="output/ltsm_textprompt_lr${lr}_loraFalse_down${downsample_rate}_freeze${freeze}_e${epoch}_pred${pred_len}/"
    CUDA_VISIBLE_DEVICES=0,1,2,3 python3 main_ltsm.py \
    --model LTSM_WordPrompt \
    --model_name_or_path gpt2-medium \
    --train_epochs ${epoch} \
    --batch_size 10 \
    --pred_len ${pred_len} \
    --gradient_accumulation_steps 64 \
    --data_path ${TRAIN} \
    --test_data_path_list ${TEST} \
    --prompt_data_path ${PROMPT} \
    --freeze ${freeze} \
    --learning_rate ${lr} \
    --downsample_rate ${downsample_rate} \
    --output_dir ${OUTPUT_PATH} \
    --eval 0
done
