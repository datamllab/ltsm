nohup bash -c '
declare -a data_paths=(
  "../../datasets/ETT-small/ETTh1.csv"
  "../../datasets/ETT-small/ETTh2.csv"
  "../../datasets/ETT-small/ETTm1.csv"
  "../../datasets/ETT-small/ETTm2.csv"
  "../../datasets/electricity/electricity.csv"
  "../../datasets/traffic/traffic.csv"
  "../../datasets/exchange_rate/exchange_rate.csv" 
  "../../datasets/weather/weather.csv"
)

declare -a data=(
  "ETTh1"
  "ETTh2"
  "ETTm1"
  "ETTm2"
  "custom"
  "custom"
  "custom"
  "custom"
)

declare -a features=(7 7 7 7 321 862 8 21)

declare -a batch_sizes=(128 128 128 128 32 24 128 128)

for index in "${!data_paths[@]}";
do
  CUDA_VISIBLE_DEVICES=0,1,2,3 python3 main_ltsm.py \
  --config "patchtst.json" \
  --data_path ${data_paths[$index]} \
  --data ${data[$index]} \
  --enc_in ${features[$index]} \
  --batch_size ${batch_sizes[$index]}
done
' > output.log 2>&1 &
echo $! > save_pid.txt