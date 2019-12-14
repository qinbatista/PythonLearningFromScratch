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
# %%
