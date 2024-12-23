#!/bin/bash
chmod +x val.sh
# 設定 Python 腳本的路徑
PYTHON_SCRIPT="/home/rfa/NTUST/LAB/A-Yolom/ultralytics/val_plot.py"
RESULTS_FILE="/home/rfa/NTUST/LAB/A-Yolom/results.csv"
MAX_RETRIES=20  # 最大重試次數
RETRY_COUNT=0
SLEEP_TIME=30  # 重試前等待的秒數

# 檢查 results.csv 是否完整的函數
check_results() {
    if [ ! -f "$RESULTS_FILE" ]; then
        echo "Results file does not exist"
        return 1
    fi

    # 計算檔案行數（扣除標題行）
    local lines=$(wc -l < "$RESULTS_FILE")
    lines=$((lines - 1))
    
    # 預期應該有13行數據（1個原始模型 + 12次迭代）
    if [ "$lines" -eq 13 ]; then
        echo "Results file is complete with $lines data rows"
        return 0
    else
        echo "Results file is incomplete with only $lines data rows"
        return 1
    fi
}

# 執行 Python 腳本的函數
run_script() {
    echo "Starting Python script..."
    python3 "$PYTHON_SCRIPT"
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "Python script completed successfully"
        return 0
    else
        echo "Python script failed with exit code $exit_code"
        return 1
    fi
}

# 主要執行邏輯
while true; do
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "Reached maximum retry attempts ($MAX_RETRIES)"
        exit 1
    fi

    # 檢查結果檔是否完整
    if check_results; then
        echo "Validation completed successfully"
        exit 0
    fi

    # 增加重試計數
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "Attempt $RETRY_COUNT of $MAX_RETRIES"
    
    # 執行 Python 腳本
    run_script
    
    # 如果腳本被 killed 或失敗，等待一段時間後重試
    if [ $? -ne 0 ]; then
        echo "Script was interrupted or failed"
        echo "Waiting $SLEEP_TIME seconds before retry..."
        sleep $SLEEP_TIME
        continue
    fi
    
    # 最後檢查一次結果
    if check_results; then
        echo "Validation completed successfully"
        exit 0
    fi
done
