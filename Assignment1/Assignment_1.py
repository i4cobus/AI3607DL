import paddle
import numpy as np
from matplotlib import pyplot as plt
from paddle.fluid.dataloader import batch_sampler
from paddle.fluid.dataloader.batch_sampler import BatchSampler
import paddle.nn.functional as F
from paddle.nn import Linear
from paddle.io import Dataset
import math

# Define 

num_samples=1000
# gauss function:
epochs=200
# # polynominal functon:
# epochs=200
batchs=400

def f(x, mean=0, sigma=1):
    return np.exp(-1*((x-mean)**2)/(2*(sigma**2)))/(math.sqrt(2*np.pi)*sigma)

# def f(x, a=1, b=-2.4, c=4.8, d=0):
#     return a*x**3+b*x**2+c*x+d

# Data
x=np.zeros(num_samples)
y=np.zeros(num_samples)

for i in range(num_samples):
    x[i]=np.random.uniform(-3.0, 3.0)
    y[i]=f(x[i])

x=paddle.to_tensor(x, dtype='float32')
y=paddle.to_tensor(y, dtype='float32')

# Multi-Layer Perceptron

class MLP(paddle.nn.Layer):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1=Linear(2, 32)
        self.fc2=Linear(32, 2)
    def forward(self, inputs):
        x=self.fc1(inputs)
        x=F.relu(x)
        x=self.fc2(x)
        return x

# Training

def train(model):
    # gauss function:
    opt=paddle.optimizer.SGD(learning_rate=0.1, parameters=model.parameters())
    # # polynominal function:
    # opt=paddle.optimizer.SGD(learning_rate=0.001, parameters=model.parameters())
    y_graph=[]
    x_graph=[]
    for i in range(epochs):
        for j in range(batchs):
            x_train=x[j*2: 2+j*2]
            y_train=y[j*2: 2+j*2]
            y_pred=model(x_train)
            if i==(epochs-1):
                y_graph.append(y_pred)
                x_graph.append(x_train)
            loss=F.square_error_cost(y_pred, y_train)
            avg_loss=paddle.mean(loss)
            if i%10==0:
                print("epoch: {},batch: {}, loss: {}".format(i, j, avg_loss.numpy()))
            avg_loss.backward()
            opt.step()
            opt.clear_grad()
    y_graph=np.array(y_graph)
    x_graph=np.array(x_graph)
    plt.plot(x_graph, y_graph, 'r.')
    x_origin=x[0:800]
    x_origin=np.array(x_origin)
    y_origin=y[0:800]
    y_origin=np.array(y_origin)
    plt.plot(x_origin,y_origin, 'b.')
    plt.show()
    paddle.save(model.state_dict(), 'MLP_test.pdparams')

model=MLP()
train(model)

# Evaluation

def evaluation(model):
    print('start evaluation .......')
    params_file_path = 'MLP_test.pdparams'
    param_dict = paddle.load(params_file_path)
    model.load_dict(param_dict)
    model.eval()
    y_graph=[]
    x_graph=[]
    for i in range(100):
        x_test=x[800+i*2: 800+2+i*2]
        y_test=y[800+i*2: 800+2+i*2]
        y_pred=model(x_test)
        y_graph.append(y_pred)
        x_graph.append(x_test)
    loss = F.square_error_cost(y_pred, y_test)
    avg_loss = paddle.mean(loss)  
    print('loss={}'.format(avg_loss.numpy()))
    y_graph=np.array(y_graph)
    x_graph=np.array(x_graph)
    plt.plot(x_graph, y_graph, 'r.')
    x_origin=x[800:1000]
    x_origin=np.array(x_origin)
    y_origin=y[800:1000]
    y_origin=np.array(y_origin)
    plt.plot(x_origin,y_origin, 'b.')
    plt.show()

evaluation(model)
