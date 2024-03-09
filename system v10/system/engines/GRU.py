import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torch.optim as optim

def getdata(country="Afghanistan",feature_name="oil_price",start=1934,tail=2014):
    '''
    返回: data，维度是(总的时间步长, 特征数量)，其中特征数量等于一个时间步里面的向量长度；
    '''
    data = pd.read_csv('./system/dataset/data.csv')
    x = data[(start<=data['year']) & (data['year']<=tail)&(data['country'] == country)]

    #x_train,x_test = split_train_test(x,0.8)
    return np.array(x['year']),np.array(x[feature_name]).reshape(-1,1)

class RNNModel(nn.Module):
    #循环神经网络模型
    def __init__(self,input_size,num_hiddens,num_directions = 1, num_layers = 1):
        super(RNNModel,self).__init__()
        self.rnn = nn.GRU(input_size, num_hiddens,num_layers,batch_first=True)
        for p in self.rnn.parameters():
            nn.init.normal_(p, mean=0.0, std=0.001)
        self.input_size = input_size
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.num_directions = num_directions
        self.output = nn.Linear(self.num_hiddens,self.input_size)#设计输出层

    def forward(self,X,state): 
        Y,state = self.rnn(X,state) 
        Y = Y.reshape(-1,self.num_hiddens)
        output = self.output(Y)
        return output,state
    
    def begin_state(self,batch_size = 1):
        return torch.zeros(size = (self.num_layers*self.num_directions,batch_size,self.num_hiddens))

#模型预测
def predict_RNN(net,x,num_steps,num_preds=0,input_size = 1):
    # x_test默认是一个一维len_x的时间序列，转换成batch_size * len_x * input_size = 1*len_x*1的tensor，是预测时候已知的前置信息；
    # num_preds，预测多少个时间段的信息，即在x后生成num_preds个时间步；
    # 思路：每次用训练样本等长的x_test片段来预测新的等长预测，
    #预测x_test内部的时间步时，只更新state，;预测然后只取第一个时间步更新到输入样本中
    n = len(x) #时间长度
    times = num_preds //num_steps + 1
    #需要预测的时间长度
    x = torch.tensor(x, dtype = torch.float32).reshape(1,-1,input_size)
    new_x = torch.zeros((n+num_preds,input_size)).reshape(1,-1,input_size)
    #num_steps = min(n,num_steps) #选择前置信息和
    y_test = torch.zeros(size=(1,n+times*num_steps,input_size))
    state = net.begin_state(batch_size = 1)
    for i in range(n-num_steps): #x_test内部预测, 3 1 2 3 4 5
        yhat,state = net(x[:,i:i+num_steps,:],state.detach()) # yhat的size是(batch_size*num_steps, input_size)
        #y_test[0,i+num_steps,:] = x[0,i+num_steps,:]
        y_test[0,i+num_steps,:] = yhat[-1,:]
        new_x[0,i+num_steps,:] = x[0,i+num_steps,:]
    
    for i in range(0,num_preds):
        yhat,state = net(new_x[:,n-num_steps+i:n+i,:],state.detach())
        y_test[0,n+i,:] = yhat[-1,:]
        new_x[0,n+i,:] = yhat[-1,:]
    return y_test


######模型预测#########
def predict(net,x,num_steps,num_preds=0,input_size = 1):
    # x_test默认是一个一维len_x的时间序列，转换成batch_size * len_x * input_size = 1*len_x*1的tensor，是预测时候已知的前置信息；
    # num_preds，预测多少个时间段的信息，即在x后生成num_preds个时间步；
    # 思路：每次用训练样本等长的x_test片段来预测新的等长预测，
    #预测x_test内部的时间步时，只更新state，;预测然后只取第一个时间步更新到输入样本中
    n = len(x) #时间长度
    x = torch.tensor(x, dtype = torch.float32).reshape(1,-1,input_size)
    num_steps = min(n,num_steps) #选择前置信息和
    y_test = torch.zeros(x.shape)
    state = net.begin_state(batch_size = 1)
    for i in range(n-num_steps): #x_test内部预测, 3 1 2 3 4 5
        yhat,state = net(x[:,i:i+num_steps,:],state.detach()) # yhat的size是(batch_size*num_steps, input_size)
        y_test[0,i+num_steps] = yhat[0]
    
    return y_test


##########可视化##########
def vis_data(year,x,output,train_test_rate= None):
    #可视化图形
    
    fig = plt.figure(figsize = (10,8))
    plt.plot(year,x,label='Real data') #画原数据曲线
    plt.plot(year,output.detach().view(-1),label='Fitted data') #画拟合曲线
    
    plt.xlabel('time')
    plt.ylabel('value')
    plt.legend()
    
    
    if train_test_rate: 
        #画辅助线
        ind = int(len(x)*train_test_rate)
        plt.axvline(x=year[ind], color='r', linestyle='--') #画辅助线
        # 在x轴顶部对应比例位置添加文本
        text_y_position = plt.gca().get_ylim()[1]  # 文本所在y轴位置
        plt.text((year[0]+year[ind])//2, text_y_position, 'Train set', ha='center',color="red",fontsize=14)
        plt.text((year[-1]+year[ind])//2, text_y_position, 'Test set', ha='center',color="red",fontsize=14)
    plt.title('Result of GRU',pad=20,fontsize=15)
    return fig
    

def GRU_model(country,feature):
    year,x = getdata(country,feature,1932,2014)
    num_steps = 5
    train_test_rate=0.9
    model_file = 'GRU'+"-"+country+'-'+feature+'.pth' 
    net = torch.load('./system/engines/model/GRU/'+model_file)
    output = predict_RNN(net,x,num_steps,num_preds=0,input_size = 1)
    fig = vis_data(year[num_steps:],x[num_steps:],output.reshape(-1)[num_steps:len(year)].detach(),train_test_rate)
    return fig

def RNN_model(country,feature):
    year,x = getdata(country,feature,1932,2014)
    num_steps = 5
    train_test_rate=0.9
    model_file = 'RNN'+"-"+country+'-'+feature+'.pth' 
    net = torch.load('./system/engines/model/RNN/'+model_file)
    output = predict_RNN(net,x,num_steps,num_preds=0,input_size = 1)
    fig = vis_data(year[num_steps:],x[num_steps:],output.reshape(-1)[num_steps:len(year)].detach(),train_test_rate)
    return fig