#Importing all the required libraries
#pip install numpy, pandas, matplotlib,s eaborn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# You can write code above the if-main block.


# if __name__ == '__main__':
#     # You should not modify this part.
#     import argparse


#     parser = argparse.ArgumentParser()
#     parser.add_argument('--training',
#                        default='training_data.csv',
#                        help='input training data file name')
#     parser.add_argument('--testing',
#                         default='testing_data.csv',
#                         help='input testing data file name')
#     parser.add_argument('--output',
#                         default='output.csv',
#                         help='output file name')
#     args = parser.parse_args()
    
#     # The following part is an example.
#     # You can modify it at will.
#     training_data = load_data(args.training)
#     trader = Trader()
#     trader.train(training_data)
    
#     testing_data = load_data(args.testing)

#     with open(args.output, 'w') as output_file:
#         for row in testing_data:
#             # We will perform your action as the open price in the next day.
#             action = trader.predict_action(row)
#             output_file.write(action)


#             # this is your option, you can leave it empty.
#             trader.re_training(i)




# Importing training dataset
training_data = pd.read_csv('training_data.csv',names=['open','high','low','close'] )
df = pd.DataFrame(training_data)

# 收盤價
close = df.reset_index()['close']

# 訓練資料每19筆資料當x, 第20筆資料當y
time_step = 19

x, y = [], []

for i in range(len(close)-time_step-1):
    x.append(close[i:(i+time_step)])
    y.append(close[(i+time_step)])

x = np.array(x)
y = np.array(y)



#now lets split data in test train pairs , 訓練資料的30%當作測試資料
# X_train 70% 842組, X_test 30% 362組, y_train 70% 842組, y_test 30% 362組

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, shuffle=False)

#調整陣列方向
x_train_ = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test_ = x_test.reshape(x_test.shape[0],x_test.shape[1],1)


from tensorflow.keras import Sequential,utils
from tensorflow.keras.layers import Flatten, Dense, Conv1D, MaxPool1D, Dropout

#使用CNN方法訓練
def regCNN():
    
    model = Sequential()
    
    model.add(Conv1D(32, kernel_size=(3,), padding='same', activation='relu', input_shape = (x_train.shape[1],1)))
    model.add(Conv1D(64, kernel_size=(3,), padding='same', activation='relu'))
    model.add(Conv1D(128, kernel_size=(5,), padding='same', activation='relu'))
    
    model.add(Flatten())
    
    model.add(Dense(50, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(units = 1))
    
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    return model


# Model Training
model_CNN = regCNN()
model_CNN.fit(x_train_, y_train, epochs=200, validation_split=0.2)

# Prediction

y_pred = model_CNN.predict(x_test_) # 30%測試資料經訓練模組得到預測的30%資料

predict = y_pred[-1] #訓練資料最後一筆測試資料的預測

print('predict:' ,predict)

# Importing test dataset
test_data = pd.read_csv('testing_data.csv' , names=['open','high','low','close'] )
df1 = pd.DataFrame(test_data)


#如果最後一筆predict  > 最後一筆close資料 則買進 +1
#如果最後一筆predict  < 最後一筆close資料 則賣出 -1

# 1 → 表示您持有 1 個單位。
# 0 → 表示您沒有持有任何單位。
# -1 → 表示您縮寫為 1 個單位。



# 開始讀取一筆正式測試資料
#


#輸出文件應命名為output.csv

    # with open(args.output, 'w') as output_file:
    #     for row in testing_data:
    #         # We will perform your action as the open price in the next day.
    #         action = trader.predict_action(row)
    #         output_file.write(action)


    #         # this is your option, you can leave it empty.
    #         trader.re_training(i)


