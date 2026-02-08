import sklearn
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("Sklearn verion is {}".format(sklearn.__version__))

# 导入数据
df = pd.read_csv('data.csv')
#  数据分析
num_size = 0.7                       # 训练集占数据集比例
outdim = 1                         # 最后一列为输出
num_samples = df.shape[0]                  # 样本个数
random_indices = np.random.permutation(num_samples)     # 生成随机排列的索引
df = df.iloc[random_indices, :]               # 根据随机排列的索引打乱数据集（不希望打乱时，注释该行）
num_train_s = round(num_size * num_samples)        # 训练集样本个数
f_ = df.shape[1] - outdim                 # 输入特征维度

#  划分训练集和测试集
P_train = df.iloc[:num_train_s, :f_]  # 训练集特征：取前num_train_s行，前f_列（所有特征列）
p_train = np.array(P_train)           # 转换为numpy数组（sklearn模型只接受数组输入）
T_train = df.iloc[:num_train_s, f_:]  # 训练集标签：取前num_train_s行，从f_列到最后一列（输出列）
T_train_flattened = np.squeeze(T_train)  # 压缩维度（比如把(100,1)的数组转为(100,)，适配模型输入）
t_train = T_train_flattened.to_numpy()   # 转换为numpy数组
M = P_train.T.shape[1]  # 训练集样本数（P_train转置后列数=原行数，等价于num_train_s）

P_test = np.array(df.iloc[num_train_s:,:f_])  # 测试集特征：取num_train_s行之后的所有行，前f_列
p_test = np.array(P_test)
T_test = df.iloc[num_train_s:, f_:]  # 测试集标签：取num_train_s行之后的所有行，输出列
T_test_flattened = np.squeeze(T_test)
t_test = T_test_flattened.to_numpy()
N = P_test.T.shape[1]  # 测试集样本数

# 创建模型
clf = make_pipeline(StandardScaler(),svm.SVR(kernel='rbf')) # StandardScaler()自动实现标准化 kernel='li'
clf.fit(p_train, t_train)

# 获取相关参数值
score_train = clf.score(p_train, t_train)
print("在训练集上的得分：", score_train)

score_test = clf.score(P_test, t_test)
print("在测试集上的得分：", score_test)

# 预测
predict1 = clf.predict(p_train)
print("预测结果：", predict1)
predict2 = clf.predict(p_test)
print("预测结果：", predict2)

#  绘图
plt.plot(range(0, M), t_train, 'r-*',linewidth=1, label='train_real')
plt.plot(range(0, M), predict1, 'b-o',linewidth=1,label='train_predict')
plt.legend('train_real', 'plt.legend()')
plt.xlabel('Sample projections')
plt.ylabel('Results of projected')
string = "SVM score_train is {}".format(score_train)
plt.title(string)
plt.xlim([-3, M+1])
plt.legend()
# 显示图形
plt.grid()
plt.show()

#  绘图
plt.plot(range(0, N), t_test, 'r-*',linewidth=1, label='test_real')
plt.plot(range(0, N), predict2, 'b-o',linewidth=1,label='test_predict')
plt.legend('test_real', 'test_predict')
plt.xlabel('Sample projections')
plt.ylabel('Results of projected')
string = "SVM score_test is {}".format(score_test)
plt.title(string)
plt.xlim([-3, N+1])
plt.legend()
# 显示图形
plt.grid()
plt.show()

# 计算相关评价指标
# R^2就等于 内置的score()函数
# 用来衡量模型拟合数据的程度，取值范围在0到1之间。R2越接近1，说明模型对数据的拟合度越高
R1 = 1 - np.linalg.norm(t_train - predict1) ** 2 / np.linalg.norm(t_train - np.mean(t_train)) ** 2 #训练集的R^2
R2 = 1 - np.linalg.norm(t_test - predict2) ** 2 / np.linalg.norm(t_test - np.mean(t_test)) ** 2 #测试集的R^2

# MAE
# 预测值与实际值之间差值的平均绝对值。MAE越小，说明模型的预测精度越高
MAE1 = np.sum(np.abs(predict1 - t_train)) / M #训练集的MAE
MAE2 = np.sum(np.abs(predict2 - t_test)) / N  #测试集的MAE

# MBE
# MBE是预测值与实际值之间差值的平均值。MBE为0表示模型的预测结果没有偏差，否则表示存在偏差
MBE1 = np.sum(predict1 - t_train) / M #训练集的MBE
MBE2 = np.sum(predict2 - t_test) / N #测试集的MBE

# MAPE
# MAPE是预测值与实际值之间百分比差值的平均绝对值
MAPE1 = np.sum(np.abs((predict1 - t_train) / t_train)) / M #训练集的MAPE
MAPE2 = np.sum(np.abs((predict2 - t_test) / t_test)) / N    #测试集的MAPE

# RMSE
# RMSE是预测值与实际值之间差值的平方的均值的平方根。RMSE越小，说明模型的预测精度越高
RMSE1 = np.sqrt(np.sum((predict1 - t_train) ** 2) / M)  # 训练集的RMSE
RMSE2 = np.sqrt(np.sum((predict2 - t_test) ** 2) / N)  # 训练集的RMSE

print("训练集数据的R2为：{}".format(R1))
print("测试集数据的R2为：{}".format(R2))
print("训练集数据的MAPE为：{}".format(MAPE1))
print("测试集数据的MAPE为：{}".format(MAPE2))
print("训练集数据的MAE为：{}".format(MAE1))
print("测试集数据的MAE为：{}".format(MAE2))
print("训练集数据的MBE为：{}".format(MBE1))
print("测试集数据的MBE为：{}".format(MBE2))
print("训练集数据的RMSE为：{}".format(RMSE1))
print("测试集数据的RMSE为：{}".format(RMSE2))

#  绘图
# 散点图
sz = 25
c = 'b'

plt.figure()
plt.scatter(t_train, predict1, sz, c)
plt.plot(plt.xlim(), plt.ylim(), '--k')
plt.xlabel('True value of training set')
plt.ylabel('Predict value of training set')
plt.xlim([min(t_train), max(t_train)])
plt.ylim([min(predict1), max(predict1)])
plt.title('True vs. Predict')
plt.show()

plt.figure()
plt.scatter(t_test, predict2, sz, c)
plt.plot(plt.xlim(), plt.ylim(), '--k')
plt.xlabel('True value of test set')
plt.ylabel('Predict value of test set')
plt.xlim([min(t_test), max(t_test)])
plt.ylim([min(predict2), max(predict2)])
plt.title('True vs. Predict')
plt.show()

# 导入带预测的数据(未来数据)
df2 = pd.read_csv('data_predict.csv')
V = df2.iloc[:, :f_].to_numpy()
predict3 = clf.predict(V)
print("预测结果：", predict3)
data = {'预测结果': predict3}
df3 = pd.DataFrame(data)
df3.to_csv('output.csv', index=False)

import joblib

joblib.dump(clf, "svr_pipeline.joblib")
# 加载模型
# clf = joblib.load("svr_pipeline.joblib")