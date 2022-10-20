# Importing all the required libraries

# Importing training dataset

# 收盤價 close[] 1224組資料

# 訓練資料每19筆資料當x, 第20筆資料當y 

# now lets split data in test train pairs , 訓練資料的30%當作測試資料
# X_train 70% 842組, X_test 30% 362組, y_train 70% 842組, y_test 30% 362組
# 利用 sklearn 調整陣列方向 

# 利用函數 tensorflow.keras 模組
# 使用CNN方法訓練

# Model Training 跑200次

# 30%測試資料經訓練模組得到預測的30%資料，訓練資料最後一筆測試資料的預測

# 如果最後一筆predict  > 最後一筆close資料 則買進 +1
# 如果最後一筆predict  < 最後一筆close資料 則賣出 -1

# 1 → 表示您持有 1 個單位。
# 0 → 表示您沒有持有任何單位。
# -1 → 表示您縮寫為 1 個單位。



# 開始讀取一筆正式測試資料
#
# 輸出文件應命名為output.csv
