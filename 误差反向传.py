#%% 创建乘法层
class MulLayer:
	def __init__(self):
		self.x = None
		self.y = None

	def forward(self,x,y):
		self.x = x
		self.y = y
		out = x * y
		return out

	def backward(self, dout):
		dx = dout * self.y
		dy = dout * self.x
		return dx, dy
#%% 创建加法层
class AddLayer(object):
	def __init__(self):
		pass

	def forward(self,x,y):
		out = x + y
		return out

	def backward(self, dout):
		dx = dout * 1
		dy = dout * 1
		return dx, dy
#%%乘法层价格正推
apple = 100
apple_num = 2
tax = 1.1
#layer
mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()
#forward
apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)
print(price)
# %%乘法层价格反推
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)
print(dapple, dapple_price, dtax)
# %%综合例子
apple = 100
apple_num = 2
orange = 150
organe_num = 3
tax = 1.1
#layer
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_organe_layer = AddLayer()
mul_tax_layer = MulLayer()

#forward
apple_price = mul_apple_layer.forward(apple, apple_num)
organe_price = mul_orange_layer.forward(organe, organe_num)
all_price = add_apple_organe_layer.forward(apple_price, organe_price)
price = mul_tax_layer.forward(all_price, tax)
print(price)
#backward
dprice = 1
dall_price,dtax = mul_tax_layer.backward(dprice)
dapple_price, dorange_price = add_apple_organe_layer.backward(dall_price)
dorange, dorange_num = mul_orange_layer.backward(dorange_price)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)
print(dapple_num, dapple, dorange, dorange_num, dtax)
# %%ReLU层类
import numpy as np
class Relu:
	def __init__(self):
		self.mask = None

	def forward(self, x):
		self.mask = (x<=0)
		out = x.copy()
		out[self.mask] = 0
		return out

	def backward(self, dout):
		dout[self.mask] = 0
		dx = dout
		return dx
#%% sigmod层
class Sigmod:
	def __init__(self):
		self.out = None

	def forward(self, x):
		out = 1 / (1 + np.exp(-x))
		self.out = out
		return out

	def backward(self, dout):
		dx = dout * (1.0 - self.out) * self.out
		return dx
#%% Affine层
class Affine:
	def __init__(self, W, b):
		self.W = W
		self.b = b
		self.x = None
		self.dW = None
		self.db = None

	def forward(self, x):
		self.x = x
		out = np.dot(x, self.W) + self.b
		return out

	def backward(self, dout):
		dx = np.dot(dout, self.W.T)
		self.dW = np.dot(self.x.T, dout)
		self.db = np.sum(dout, axis=0)
		return dx
#%%Softmax-with-Loss 层设计
def softmax(a):
	c = np.max(a)
	exp_a = np.exp(a-c)
	sum_exp_a = np.sum(exp_a)
	y = exp_a/sum_exp_a
	return y
def cross_entropy_error(y,t):
	delta = 1e-7
	return -np.sum(t*np.log(y + delta))
class SoftmaxWithLoss:
	def __init__(self):
		self.loss = None
		self.y = None
		self.t = None

	def forward(self, x, t):
		self.t = t
		self.y = softmax(x)
		self.loss = cross_entropy_error(self.y, self.t)
		return self.loss

	def backward(self, dout=1):
		batch_size = self.t.shape[0]
		dx = (self.y - self.t) / batch_size
		return dx


# %%
import numpy as np
x = np.array([[1,2,3,4],[3,4,5,6],[5,6,7,8],[7,8,9,10]])
print(x[0])
print(x[0][0])
print(x[0:3,0:2])
# %%
