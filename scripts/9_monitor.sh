#!/bin/bash
# cpu_gpu_simple.sh - Jetson Nano 内存/CPU/GPU 简洁监控 + 彩色高亮

echo "简洁模式监控 RAM/CPU/GPU 利用率和温度 (按 Ctrl+C 退出)"

# ANSI 颜色
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # no color

sudo tegrastats | while read -r line
do
    timestamp=$(date '+%H:%M:%S')

    # --------------------
    # 内存
    # --------------------
    # 提取 RAM 使用和总量
    ram=$(echo "$line" | grep -o -E "RAM [0-9]+/[0-9]+MB")
    if [[ -n $ram ]]; then
        # 先去掉 "RAM " 前缀和 "MB" 后缀
        ram_clean=${ram#RAM }       # 去掉前缀 "RAM "
        ram_clean=${ram_clean%MB}   # 去掉后缀 "MB"
        # 以 / 分割
        ram_used=${ram_clean%%/*}   # / 前
        ram_total=${ram_clean##*/}  # / 后
    else
        ram_used=0
        ram_total=4096  # 默认 4GB
    fi

    ram_perc=$(( ram_used * 100 / ram_total ))

    # 内存颜色判断
    if [ $ram_perc -ge 80 ]; then ram_col=$RED; else ram_col=$GREEN; fi

    # --------------------
    # CPU 利用率
    # --------------------
    # 提取 CPU 各核利用率百分比
    cpu=$(echo "$line" | grep -o "CPU \[[^]]*\]" | sed 's/CPU \[\(.*\)\]/\1/' | sed 's/@[0-9]\+//g')

    # 转换为数字数组
    cpu_nums=($(echo $cpu | tr ' ' '\n' | sed 's/%//g'))

    # 计算平均 CPU 利用率
    cpu_avg=$(echo "${cpu_nums[@]}" | awk '{sum=0; for(i=1;i<=NF;i++) sum+=$i; printf "%d", sum/NF}')

    # 计算单核最大利用率
    cpu_max=$(echo "${cpu_nums[@]}" | awk '{max=$1; for(i=2;i<=NF;i++) if($i>max) max=$i; printf "%d", max}')

    # 彩色判断
    if [ $cpu_avg -ge 80 ]; then cpu_avg_col=$RED; else cpu_avg_col=$GREEN; fi
    if [ $cpu_max -ge 80 ]; then cpu_max_col=$RED; else cpu_max_col=$GREEN; fi

    # --------------------
    # GPU 利用率
    # --------------------
    # 提取 GPU 利用率百分比
    gpu=$(echo "$line" | grep -o "GR3D_FREQ [0-9]\+%" | awk '{gsub(/%/,"",$2); print $2}')
    if [ $gpu -ge 80 ]; then gpu_col=$RED; else gpu_col=$GREEN; fi

    # --------------------
    # CPU 温度
    # --------------------
    cpu_temp=$(echo "$line" | grep -o -E "CPU@[0-9.]+C" | cut -d'@' -f2 | cut -d'C' -f1)
    [ -z "$cpu_temp" ] && cpu_temp=0
    cpu_temp_col=$GREEN
    (( $(echo "$cpu_temp >= 80" | bc -l) )) && cpu_temp_col=$RED

    # --------------------
    # GPU 温度
    # --------------------
    gpu_temp=$(echo "$line" | grep -o -E "GPU@[0-9.]+C" | cut -d'@' -f2 | cut -d'C' -f1)
    [ -z "$gpu_temp" ] && gpu_temp=0
    gpu_temp_col=$GREEN
    (( $(echo "$gpu_temp >= 80" | bc -l) )) && gpu_temp_col=$RED

    # --------------------
    # 输出
    # --------------------
    echo -e "[$timestamp] RAM: ${ram_col}${ram_perc}%${NC} | CPU: $cpu | AVG: ${cpu_avg_col}${cpu_avg}%${NC} | MAX: ${cpu_max_col}${cpu_max}%${NC} | GPU: ${gpu_col}${gpu}%${NC} | CPU_TEMP: ${cpu_temp_col}${cpu_temp}C${NC} | GPU_TEMP: ${gpu_temp_col}${gpu_temp}C${NC}"
done
