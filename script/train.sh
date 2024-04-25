#!/bin/bash
# shellcheck disable=SC2086

# 激活conda环境
echo "Activating conda environment..."
source  /home/zdd/miniconda3/bin/activate pytorch2.0.1
path="/mnt/h/EEG/CAW-MASA-STST"

# 定义参数
# select_classes = [6, 72]
params_list=(
    "--num_class 6"
    "--num_class 72"
)

# 遍历参数列表
for params in "${params_list[@]}"; do

    # 检查GPU显存
    while true; do
        python $path/script/check_gpu.py
        if [ $? -eq 0 ]; then
            break  # if GPU memory is enough, break the loop and start training
        else
            echo "GPU memory is not enough, waiting..."
            sleep 300  # wait for half an hour
        fi
    done

    # 执行训练并将PID添加到进程数组
    output_file=$path"/outputs/out$(echo "$params" | sed 's/--/-/g' | sed 's/ /=/g').txt"  # 使用参数创建文件名
    mkdir -p "$(dirname "$output_file")"  # 创建文件所在的目录
    python $path/main.py $params > $output_file &
    echo "PID: $!, Parameters starting: $params"  # 打印出开始训练的参数

done
echo "All processes finished."  # 打印出所有进程结束的信息