#乘法层实现
class MulLayer:
    def __init__(self):
        self.x=None
        self.y=None
    def forward(self,x,y):
        self.x=x
        self.y=y
        out=x*y
        return out
    def backward(self,dout):
        dx=dout*self.y
        dy=dout*self.x
        return dx,dy
#加法层实现
class AddLayer:
    def __init__(self):
        pass
    def forward(self,x,y):
        out=x+y
        return out
    def backward(self,dout):
        dx=dout*1
        dy=dout*1
        return dx,dy
'''
#乘法层测试
apple=100
apple_num=2
tax=1.1
#定义乘法层
mul_apple_layer=MulLayer()
mul_tax_layer=MulLayer()
#前向计算
apple_price=mul_apple_layer.forward(apple,apple_num)
price=mul_tax_layer.forward(apple_price,tax)
print(price)
#反向求导
dprice=1
dapple_price,dtax=mul_tax_layer.backward(dprice)
dapple,dapple_num=mul_apple_layer.backward(dapple_price)
print(dapple,dapple_num,dtax)
'''


'''
#加法层+乘法层测试
apple=100
apple_num=2
orange=150
orange_num=3
tax=1.1
#定义层
mul_apple_layer=MulLayer()
mul_orange_layer=MulLayer()
add_apple_orange_layer=AddLayer()
mul_tax_layer=MulLayer()
#前向计算
apple_price=mul_apple_layer.forward(apple,apple_num)
orange_price=mul_orange_layer.forward(orange,orange_num)
all_price=add_apple_orange_layer.forward(apple_price,orange_price)
price=mul_tax_layer.forward(all_price,tax)
#反向求导
dprice=1
dall_price,d_tax=mul_tax_layer.backward(dprice)
dapple_price,dorange_price=add_apple_orange_layer.backward(dall_price)
dorange,dorange_num=mul_orange_layer.backward(dorange_price)
dapple,dapple_num=mul_apple_layer.backward(dapple_price)
print(price)
print(dapple_num,dapple,dorange,dorange_num,d_tax)
'''