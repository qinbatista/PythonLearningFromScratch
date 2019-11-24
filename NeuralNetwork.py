#%% 跃阶函数
import numpy as np
import matplotlib.pylab as plt

def step_function(x):
	return np.array(x>0, dtype = np.int)

x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
plt.plot(x,y)
plt.ylim(-0.1,1.1)
plt.show()

# %% sigmoid函数
def sigmoid(x):
	return 1 / (1+ np.exp(-x))
x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x,y)
plt.ylim(-0.1,1.1)
plt.show()

# %% 数组操作
A = np.array([1,2,3,4])
print(A)
print(np.ndim(A))
print(A.shape)
print(A.shape[0])

# %%矩阵点乘
A = np.array([[1,2],[3,4]])
B = np.array([[4,5],[6,7]])
np.dot(A,B)

# %% 加权值 A = XW+B
#第一层神经元
X = np.array([1.0,0.5])
W1 = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
B1 = np.array([0.1, 0.2, 0.3])
A1 = np.dot(X, W1)+B1
Z1 = sigmoid(A1)
print(f'第一层神经元权重:{Z1}')
#第二层神经元
W2 = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
B2 = np.array([0.1,0.2])
A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)
print(f'第二层神经元权重:{Z2}')

W3 = np.array([[0.1,0.3],[0.2,0.4]])
B3 = np.array([0.1,0.2])

A3 = np.dot(Z2, W3) +B3

def identify_function(x):
	return x
A3 = np.dot(Z2,W3) + B3
Y = identify_function(A3)
print(f"最后一层结果(恒等结果):{Y}")
# %% softmax计算方法
a = np.array([0.3,2.9,4.0])
exp_a = np.exp(a)
print(exp_a)

sum_exp_a = np.sum(exp_a)
print(sum_exp_a)

y = exp_a/sum_exp_a
print(y)

# %%softmax方法
def softmax(a):
	c = np.max(a)
	exp_a = np.exp(a-c)
	sum_exp_a = np.sum(exp_a)
	y = exp_a/sum_exp_a
	return y
y = softmax(Y)
print(y)
np.sum(y)
# %%
