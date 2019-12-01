#%% 跃阶函数
import numpy as np
import matplotlib.pylab as plt
import sys,os
sys.path.append(os.pardir)
from source_code.dataset.mnist import load_mnist
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
(x_train, t_train), (x_test,y_test) = load_mnist(flatten=True, normalize=False)
print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(y_test.shape)
from PIL import Image
def img_show(img):
	pil_img = Image.fromarray(np.uint8(img))
	pil_img.show()
(x_train, t_train), (x_test,y_test) = load_mnist(flatten=True, normalize=False)
img = x_train[0]
label = t_train[0]
print(label)
print(img.shape)
img = img.reshape(28,28)
print(img.shape)
img_show(img)

# %%
import pickle

def get_data():
	(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
	return x_test, t_test


def init_network():
	with open("/Users/batista/LearningProject/PythonLearningFromScratch/source_code/ch03/sample_weight.pkl", 'rb') as f:
		network = pickle.load(f)
	return network


def predict(network, x):
	W1, W2, W3 = network['W1'], network['W2'], network['W3']
	b1, b2, b3 = network['b1'], network['b2'], network['b3']
	a1 = np.dot(x, W1) + b1
	z1 = sigmoid(a1)
	a2 = np.dot(z1, W2) + b2
	z2 = sigmoid(a2)
	a3 = np.dot(z2, W3) + b3
	y = softmax(a3)
	return y

# %%
x, t = get_data()
network = init_network()

accuracy_cnt = 0
for i in range(len(x)):
	y = predict(network, x[i])
	p = np.argmax(y)
	if p == t[i]:
		accuracy_cnt+=1
print("Accuracy:"+str(float(accuracy_cnt)/len(x)))

# %%
x, t = get_data()
network = init_network()
batch_size = 100
accuracy_cnt = 0
for i in range(0, len(x), batch_size):
	x_batch = x[i:i+batch_size]
	y_batch = predict(network, x_batch)
	p = np.argmax(y_batch, axis=1)
	accuracy_cnt += np.sum(p ==t[i:i+batch_size])
print("Accuracy:"+str(float(accuracy_cnt)/len(x)))


# %%均方误差
def mean_squared_error(y, t):
	return 0.5 * np.sum((y-t)**2)

t = [0,0,1,0,0,0,0,0,0,0]
y1 = [0.1,0.05,0.6,0,0.05,0.1,0,0.1,0,0]
mean_squared_error(np.array(y1),np.array(t))
#%%
y2 = [0.1,0.05,0.1,0,0.05,0.1,0,0.6,0,0]
mean_squared_error(np.array(y2),np.array(t))


# %%交叉墒
def cross_entropy_error(y,t):
	delta = 1e-7
	return -np.sum(t*np.log(y + delta))

cross_entropy_error(np.array(y1), np.array(t))
#%%
cross_entropy_error(np.array(y2), np.array(t))

# %%
print(x_train.shape)
print(t_train.shape)

# %%
train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
print(x_batch)
t_batch = t_train[batch_mask]
print(t_batch)

#%%求导数
def numerical_diff(f,x):
	h = 1e-4
	return (f(x+h)- f(x-h))/(2*h)

#%%求y = 0.01*x**2+ 0.1*x
def function_1(x):
	return 0.01*x**2 + 0.1*x
import numpy as np
import matplotlib.pylab as plt
x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x,y)
plt.show()
# %%
print(numerical_diff(function_1,5))
print(numerical_diff(function_1,10))
# %%f(x0,x1) = x0**2+ x1**2 函数
def function_2(x):
	return x[0]**2 + x[1]**2

#%%梯度求解法
def numerical_gradient(f,x):
	h = 1e-4
	grad = np.zeros_like(x) #生成和X形状相同导数组
	for idx in range(x.size):
		tmp_val = x[idx]
		x[idx] = tmp_val+h
		fxh1 = f(x)
		x[idx] = tmp_val - h
		fxh2 = f(x)
		grad[idx] = (fxh1 - fxh2)/(2*h)
		x[idx] = tmp_val
	return grad
print(numerical_gradient(function_2,np.array([3.0,4.0])))
print(numerical_gradient(function_2,np.array([0.0,2.0])))
print(numerical_gradient(function_2,np.array([3.0,4.0])))

# %%梯度法（下降）
def gradient_descent(f , init_x, lr = 0.01, step_num =100):
	x = init_x
	for i in range(step_num):
		grad = numerical_gradient(f, x)
		x -= lr * grad
	return x
#%%梯度下降法求例子
init_x = np.array([-3.0, 4.0])
gradient_descent(function_2, init_x = init_x, lr=0.1, step_num=100)

# %%
