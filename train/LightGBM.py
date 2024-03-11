# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 15:32:20 2024

@author: echo
"""
#import packages
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import os

folder_path = "./train/model/LightGBM" 
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

def zscore_data(train_data):
    columns=train_data.columns
    for feature in columns[2:]:
        feature_mean=train_data[feature].mean()
        feature_std=train_data[feature].std()
        train_data[feature]=(train_data[feature]-feature_mean)/feature_std
    return train_data

def lgb_model_training(data,target,start_year,train_years,validation_years,subsample,learning_rate,num_leaves,num_boost,X_test):
    train_start=pd.to_datetime(str(start_year))
    train_end=train_start+pd.DateOffset(years=train_years)-pd.DateOffset(days=1)
    validation_start=train_end+pd.DateOffset(days=1)
    validation_end=validation_start+pd.DateOffset(years=validation_years)-pd.DateOffset(days=1)
    
    feature=data.columns.tolist()
    X_train=data.loc[train_start:train_end][feature].values
    y_train=target.loc[train_start:train_end].values
    X_validation=data.loc[validation_start:validation_end][feature].values
    y_validation=target.loc[validation_start:validation_end].values
    
    print(y_train)
    train_data = lgb.Dataset(X_train, label=y_train)
    
    params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mse',
    'num_leaves': num_leaves,
    'learning_rate': learning_rate,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
    }
    
    print("start model training:")
    
    lgb_reg = lgb.train(params, train_data, num_boost_round=num_boost)
    y_pred = lgb_reg.predict(X_validation)
    
    #calculate mse of validation set
    va_y=y_validation.reshape(-1,1)
    va_y_pred = y_pred.reshape(-1,1)
    mean_square_error=np.mean((va_y-va_y_pred)**2)
    
    y_pred_test=lgb_reg.predict(X_test)
    va_y_pred = np.vstack((va_y_pred, y_pred_test))
    return mean_square_error,va_y,va_y_pred,lgb_reg

def lgb_model_testing(trained_model,data,target,start_year,train_years,validation_years,subsample,X_test):
    train_start=pd.to_datetime(str(start_year))
    train_end=train_start+pd.DateOffset(years=train_years)-pd.DateOffset(days=1)
    validation_start=train_end+pd.DateOffset(days=1)
    validation_end=validation_start+pd.DateOffset(years=validation_years)-pd.DateOffset(days=1)  
    feature=data.columns.tolist()
    X_validation=data.loc[validation_start:validation_end][feature].values
    y_validation=target.loc[validation_start:validation_end].values 
    
    print("start model training:")
    lgb_reg=trained_model
    y_pred=lgb_reg.predict(X_validation)
    
    #calculate mse of validation set
    va_y=y_validation.reshape(-1,1)
    va_y_pred=y_pred.reshape(-1,1)
    mean_square_error=np.mean((va_y-va_y_pred)**2)
    
    y_pred_test=lgb_reg.predict(X_test)
    va_y_pred = np.vstack((va_y_pred, y_pred_test))
    return mean_square_error,va_y,va_y_pred
#Set year as index

def lightgbm_func(source_data,city,feature_name,train = 1):
    source_data['year'] = pd.to_datetime(source_data['year'], format='%Y')
    source_data = source_data.sort_values(by=['year'])
    source_data = source_data.set_index(['year',])
    train_data=source_data[source_data['country']==city]

    target_1=pd.DataFrame(train_data[feature_name].values,index=train_data[feature_name].index - pd.DateOffset(years=1),columns=[feature_name])
    target_1=target_1.drop(target_1.index[0])

    processed_train_data=processing_data(train_data)
    processed_train_data=zscore_data(processed_train_data)
    processed_train_data=processed_train_data.iloc[:,2:]    
    X_test = processed_train_data.iloc[-1].values
    X_test = X_test.reshape(1,-1)
    processed_train_data = processed_train_data.drop(processed_train_data.index[-1])
    
    if train == 1:
        subsample_list=[0.8,0.85,0.9]
        learning_rate_list=[0.01,0.1,0.001]
        num_leaves=[2,4,6]
        num_boosts=[7000,8000,9000]
        MSE=np.inf
        combination=[0,0,0,0]
        for x in range(0,3):
            for y in range(0,3):
                for z in range(0,3):
                    for a in range(0,3):
                        subsample=subsample_list[x]
                        learning_rate=learning_rate_list[y]
                        num_leave=num_leaves[z]
                        num_boost=num_boosts[a]
                        start_year=1932
                        train_years=int(len(processed_train_data)*subsample)
                        validation_years=len(processed_train_data)-train_years               
                        result=lgb_model_training(processed_train_data,target_1,start_year,train_years,validation_years,subsample,learning_rate,num_leave,num_boost,X_test) 
                        #find best prediction:
                        if result[0]<MSE:
                            MSE = result[0]
                            va_y = result[1]
                            va_pred = result[2]
                            used_train = train_years
                            combination[0] = x
                            combination[1] = y
                            combination[2] = z
                            combination[3] = a
                            fmodel = result[3]
        model_file ='LightGBM'+'-'+city+'-'+feature_name+'.txt'  
        model_path = os.path.join(folder_path, model_file)
        fmodel.save_model(model_path)
        print(combination)
        
    subsample=0.7
    print("subsample:",subsample)
    start_year=1932
    train_years=int(len(processed_train_data)*subsample)
    validation_years=len(processed_train_data)-train_years
     
    model_file ='LightGBM'+'-'+city+'-'+feature_name+'.txt'  
    model_path = os.path.join(folder_path, model_file)
    model = lgb.Booster(model_file=model_path)
    result = lgb_model_testing(model,processed_train_data,target_1,start_year,train_years,validation_years,subsample,X_test)
    MSE = result[0]
    va_y = result[1]
    va_pred = result[2]
    used_train = train_years

    #vasualization:
    x = range(0,len(va_pred))
    years = []
    start_year = 1933
    for i in x:
        years.append(pd.to_datetime(str(start_year+used_train))+pd.DateOffset(years=i))
    #fig,ax = plt.subplots(figsize = (10,6))
    plt.plot(years,va_pred,label = 'model test')
    plt.plot(years[:-1],va_y,label = 'real value')
    plt.xlabel('last predicted years')
    plt.ylabel('price')
    plt.legend()
    plt.title("Lightgbm Model")
    plt.show()
    print('train model totally using ',used_train,'pieces of data','mse:',MSE)
    return plt,va_pred,years
'''
source_data = pd.read_csv('data.csv')
city = 'USA'
feature_name = 'oil_price'
lightgbm_func(source_data,city, feature_name,train = 1)
'''

