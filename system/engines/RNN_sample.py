import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torch.optim as optim



#####数据准备##########
def getdata():
    '''
    返回: data，维度是(总的时间步长, 特征数量)，其中特征数量等于一个时间步里面的向量长度；
    '''
    data = pd.read_csv('./Dataset/Oil and Gas 1932-2014.csv')
    # 给定特征、城市以及区间
    city = 'Afghanistan'
    feature_name = 'oil_price_2000' 
    start = 1934
    tail = 2014
    #获取数据集合
    x = data[(start<=data['year']) & (data['year']<=tail)&(data['cty_name'] == city)]

    #x_train,x_test = split_train_test(x,0.8)
    return x['year'],np.array(x[feature_name]).reshape(-1,1)

#获取batch；
def seq_data_iter_random(x, batch_size, num_steps, num_delays=1):
    '''
    输入：x是data，维度是(总的时间步长, 特征数量)；num_steps是一个样本的时间步长，训练时是定长；num_delays是t时刻的输入返回t+num_delays时刻的输出；
    返回：生成器，生成同维的特征向量X和Y，维度是(batch_size,num_steps,input_size)
    '''
    
    x = x[random.randint(0, (num_steps - 1)):] #第一步随机裁剪序列偏移
 
    # 减去1，是因为我们需要考虑标签
    num_subseqs = (len(x) - 1) // num_steps #确定当前序列可以根据num_steps裁剪出多少个样本；

    initial_indices = list(range(0, num_subseqs * num_steps, num_steps)) #收集每个样本的第一个时间步下标作为标志
    
    random.shuffle(initial_indices) #随机打乱开头

    def data(pos):
        '''
        提取pos为标记的样本
        '''
        return x[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size #总样本数 = batch_size * num_batches

    for i in range(0, batch_size * num_batches, batch_size): #每次取一个batch
        initial_indices_per_batch = initial_indices[i: i + batch_size] #提取batch_size个样本开头
        X = np.array([data(j) for j in initial_indices_per_batch]) #将这batch_size
        Y = np.array([data(j + num_delays) for j in initial_indices_per_batch])
        yield torch.tensor(X,dtype=torch.float32), torch.tensor(Y,dtype=torch.float32)

#划分时间序列的训练集和测试集
def split_train_test(X,ratio):
    '''
    输入：x是总数据集，维度是(总的时间步长, 特征数量)；ratio:int,表示训练集和测试集的比例
    '''
    len_data = len(x)
    return X[:round(len_data*ratio)],X[round(len_data*ratio):]
    size = X.shape
    if len(size) == 1: #一维
        return X[:round(size[0]*ratio)],X[round(size[0]*ratio):]
    else: #二维，有多个变量划分
        return X[:,round(size[0]*ratio)], X[:,round(size[0]*ratio):]

#####################定义RNN模型##########################
#定义RNN模型 num_hiddens
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
    
#模型训练
def train_epoch(net,x_train,num_epochs,batch_size,num_steps,lr=0.001):
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr)
    l = [] #用来绘制损失函数
    #训练
    for epoch in range(num_epochs):
        train_iter = seq_data_iter_random(x_train,batch_size,num_steps) #一个RNN的输入长度
        for X,Y in train_iter:
            optimizer.zero_grad()
            state = net.begin_state(batch_size = batch_size)
            output,state = net(X,state)
            loss = criterion(output,Y.reshape(-1,1))
            loss.backward()
            optimizer.step()
        ##报loss
        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')
        l.append(loss.item())
        
    ##############################绘制损失函数#################################
    plt.plot(l,'r')
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    return net


##########可视化##########
def vis_data(year,x,output):
    #可视化图形
    ind = int(len(x)*train_test_rate)
    plt.figure(figsize = (10,8))
    plt.plot(year,x,label='Real data') #画原数据曲线
    plt.plot(year,output.detach().view(-1),label='Fitted data') #画拟合曲线
    plt.axvline(x=year[ind], color='r', linestyle='--') #画辅助线
    plt.xlabel('time')
    plt.ylabel('value')
    plt.legend()
    # 在x轴顶部对应比例位置添加文本
    text_y_position = plt.gca().get_ylim()[1]  # 文本所在y轴位置
    plt.text((year[0]+year[ind])//2, text_y_position, 'Train set', ha='center',color="red",fontsize=14)
    plt.text((year[-1]+year[ind])//2, text_y_position, 'Test set', ha='center',color="red",fontsize=14)
    plt.title('Result of RNN',pad=20,fontsize=15)
    #plt.text(4, 8, 'Hello, Matplotlib!', fontsize=12, color='blue', ha='center')
    #plt.text(7, 8, 'Hello, Matplotlib!', fontsize=12, color='blue', ha='center')
    plt.show()



#模型预测
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


if __name__ == "__main__":
    #定义
    num_steps = 5    # 训练时时间窗的步长
    input_size = 1          # 输入数据维度
    num_hiddens = 2560        # 隐含层维度
    output_size = 1         # 输出维度
    num_layers = 1
    num_epochs = 1000
    num_delays = 4  #时间步长的后退
    lr=0.0001
    batch_size = 6
    train_test_rate = 0.8
    
    
    #获取数据
    year,x = getdata()
    year = np.array(year)
    x_train,x_test = split_train_test(x,train_test_rate)
    
    
    #产生批
    train_iter = seq_data_iter_random(x, batch_size, num_steps, num_delays=1)
    X,Y = next(train_iter)
    print('输入批的尺寸',X.shape)
    net = RNNModel(input_size,num_hiddens)
    state = net.begin_state(batch_size)
    #训练模型
    net = train_epoch(net,x_train,num_epochs,batch_size,num_steps)
    #看输出
    Yhat,_ = net(X,state.detach())
    vis
    
    