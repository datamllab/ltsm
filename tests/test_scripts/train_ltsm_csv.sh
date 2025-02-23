nohup bash -c '
data_paths="../../datasets/ETT-small/ETTh1.csv 
../../datasets/ETT-small/ETTh2.csv 
../../datasets/ETT-small/ETTm1.csv 
../../datasets/ETT-small/ETTm2.csv 
../../datasets/electricity/electricity.csv 
../../datasets/traffic/traffic.csv 
../../datasets/exchange_rate/exchange_rate.csv 
../../datasets/weather/weather.csv"

declare -a pred_len=(96 192 336 720)

for index in "${!pred_len[@]}";
do
  CUDA_VISIBLE_DEVICES=0,1,2,3 python3 main_ltsm.py \
  --config "ltsm.json" \
  --data_path ${data_paths} \
  --test_data_path_list ${data_paths} \
  --pred_len ${pred_len[$index]} \
  --output_dir output/ltsm_lr1e-3_loraFalse_down20_freeze0_e2_pred${pred_len[$index]}/,
done
' > output.log 2>&1 &
echo $! > save_pid.txt