import inspect
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import packages
from sklearn.preprocessing import MinMaxScaler

#data cleaning functions
def processing_data(train_data):
    for column in list(train_data.columns[train_data.isnull().sum()>0]):
        mean=train_data[column].mean()
        median=train_data[column].median()
        sigma=train_data[column].std()
        train_data[column].fillna(median,inplace=True)
        feature_celling=mean+3*sigma
        feature_floor=mean-3*sigma
        train_data[column]=np.clip(train_data[column],feature_floor,feature_celling)
    return train_data


def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq ,train_label))
    return inout_seq


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(in_features =hidden_layer_size ,out_features = output_size)
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                             torch.zeros(1, 1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

def LSTM_model_training(data,learning_rate,tw=12,predicttime=12):
    print(len(data))
    scaler = MinMaxScaler(feature_range=(-1, 1))
    processed_train_data = scaler.fit_transform(data.reshape(-1, 1))
    train_data_normalized = torch.FloatTensor(processed_train_data).view(-1)
    #train_data_normalized = processed_train_data
    #[a1 a2 a3 a4 a5]->[a6]
    
    train_window = tw
    train_inout_seq = create_inout_sequences(train_data_normalized, train_window)
    
    model = LSTM()
    model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                    torch.zeros(1, 1, model.hidden_layer_size))
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
      
    epochs = 150
    for i in range(epochs):
        for seq, labels in train_inout_seq:
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                            torch.zeros(1, 1, model.hidden_layer_size))
            y_pred = model(seq)
    
            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()

    if i%25 == 1:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

    print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')
    print("start model training:")
    test_inputs = train_data_normalized[-train_window:].tolist()
    model.eval()
    
    for i in range(predicttime+2):
        seq = torch.FloatTensor(test_inputs[-train_window:])
        with torch.no_grad():
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                            torch.zeros(1, 1, model.hidden_layer_size))
            test_inputs.append(model(seq).item()) 
    pre = np.array(test_inputs[tw:])
    pre = scaler.inverse_transform(pre.reshape(-1,1))
    return pre,model

def LSTM_model_test(data,learning_rate,tw=12,predicttime=12):
    Le = len(data)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    processed_train_data = scaler.fit_transform(data.reshape(-1, 1))
    train_data_normalized = torch.FloatTensor(processed_train_data).view(-1)
    #train_data_normalized = processed_train_data
    #[a1 a2 a3 a4 a5]->[a6]
    
    train_window = tw
    
    model = LSTM()
    model.load_state_dict(torch.load('model_lstm.pth'))
    model.eval()

    test_inputs = train_data_normalized[-train_window:].tolist()
    for i in range(predicttime+2):
        seq = torch.FloatTensor(test_inputs[-train_window:])
        with torch.no_grad():
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                             torch.zeros(1, 1, model.hidden_layer_size))
            test_inputs.append(model(seq).item()) 
    pre = np.array(test_inputs[tw:])
    pre = scaler.inverse_transform(pre.reshape(-1,1))
    return pre,model

def predict(data,training = 1):
    #Set year as index
    source_data = data
    source_data['year'] = pd.to_datetime(source_data['year'], format='%Y')
    current_date = source_data['year'].values[-1]
    print(current_date)
    source_data = source_data.sort_values(by=['year'])
    source_data = source_data.set_index(['year',])
    train_data=source_data[source_data['id']=='USA']
    
    #target列设置为oil_price_2000
    target_1=pd.DataFrame(train_data['oil_price'],index=train_data['oil_price'].index,columns=['oil_price'])
    target_2=pd.DataFrame(train_data['gas_price'],index=train_data['gas_price'].index,columns=['gas_price'])
    
    #删去target列,cty_name和id
    processed_train_data=processing_data(target_1).values
    tw=[6,8,10,12]
    predicttime=[8]
    lr=[0.01,0.05]
    MSE=1000000
    combination=[0,0,0]
    for x in tw:
        for y in predicttime:
            for z in lr:
                wind=x
                pt=y
                lrate=z
                print("tw:",wind)
                print("pt:",pt)
                print("lr:",lrate)
                if training == 1:
                    result,model=LSTM_model_training(processed_train_data[:-pt],lrate,wind,pt)
                else:
                    result,model=LSTM_model_test(processed_train_data[:-pt],lrate,wind,pt)                   
                result=np.array(result)
                print(result[:-2]-np.array(processed_train_data[-pt:]))
                MSE_r =np.mean(np.sum((result[:-2]-np.array(processed_train_data[-pt:]))**2))
                #find best prediction:
                if MSE_r < MSE:
                    MSE = MSE_r
                    print(MSE_r)
                    va_y = np.array(processed_train_data[-pt:])
                    va_pred = result
                    combination[0] = x
                    combination[1] = y
                    combination[2] = z
                    fm = model  
                
    #vasualization:
    x = range(0,predicttime[0]+2)
    years = []
    for i in x[::-1]:
        years.append(current_date + pd.DateOffset(years=2) - pd.DateOffset(years=i))
    
    va_y = processed_train_data[-combination[1]:]
    plt.plot(years,va_pred,label = 'model test')
    x = range(1,len(va_pred)+1)
    plt.plot(years[:-2],va_y,label = 'real value')
    plt.xlabel('last predicted years')
    plt.ylabel('price')
    plt.legend()
    plt.title("LSTM Model")
    plt.show()
    torch.save(fm.state_dict(), 'model_lstm.pth')
    return plt,va_pred,va_y
