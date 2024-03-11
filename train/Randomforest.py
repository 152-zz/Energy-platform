# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 15:32:20 2024

@author: echo
"""
#import packages
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import matplotlib.pyplot as plt
import os

folder_path = "./train/model/Randomforest"  
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

def rf_model_training(data,target,start_year,train_years,validation_years,subsample,ne,md,mss,X_test):
    train_start=pd.to_datetime(str(start_year))
    train_end=train_start+pd.DateOffset(years=train_years)-pd.DateOffset(days=1)
    validation_start=train_end+pd.DateOffset(days=1)
    validation_end=validation_start+pd.DateOffset(years=validation_years)-pd.DateOffset(days=1)   
    feature=data.columns.tolist()
    X_train=data.loc[train_start:train_end][feature].values
    y_train=target.loc[train_start:train_end].values
    X_validation=data.loc[validation_start:validation_end][feature].values
    y_validation=target.loc[validation_start:validation_end].values
    
    print("start model training:")
    rf_regressor = RandomForestRegressor(n_estimators=ne, max_depth=md, min_samples_split=mss)
    rf_regressor.fit(X_train, y_train)
    y_pred = rf_regressor.predict(X_validation)
    
    #calculate mse of validation set
    va_y=y_validation.reshape(-1,1)
    va_y_pred = y_pred.reshape(-1,1)
    mean_square_error=np.mean((va_y-va_y_pred)**2)
    
    y_pred_test=rf_regressor.predict(X_test)
    va_y_pred = np.vstack((va_y_pred, y_pred_test))
    return mean_square_error,va_y,va_y_pred,rf_regressor

def rf_model_testing(trained_model,data,target,start_year,train_years,validation_years,subsample,X_test):
    train_start=pd.to_datetime(str(start_year))
    train_end=train_start+pd.DateOffset(years=train_years)-pd.DateOffset(days=1)
    validation_start=train_end+pd.DateOffset(days=1)
    validation_end=validation_start+pd.DateOffset(years=validation_years)-pd.DateOffset(days=1)  
    feature=data.columns.tolist()
    X_validation=data.loc[validation_start:validation_end][feature].values
    y_validation=target.loc[validation_start:validation_end].values 
    
    print("start model training:")
    rf =trained_model
    y_pred=rf.predict(X_validation)
    
    #calculate mse of validation set
    va_y=y_validation.reshape(-1,1)
    va_y_pred=y_pred.reshape(-1,1)
    mean_square_error=np.mean((va_y-va_y_pred)**2)
    
    y_pred_test=rf.predict(X_test)
    va_y_pred = np.vstack((va_y_pred, y_pred_test))
    return mean_square_error,va_y,va_y_pred
#Set year as index

def randomforest_func(source_data,city, feature_name,train = 1):
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
        subsample_list=[0.7,0.75,0.8,0.85]
        nes=[10,20,30,40,50]
        mds=[5,8,10,15]
        msss=[2,3,4,5]
        MSE=1000000
        combination=[0,0,0,0]
        for x in range(0,4):
            for y in range(0,3):
                for z in range(0,3):
                    for a in range(0,2):
                        subsample=subsample_list[x]
                        ne=nes[y]
                        md=mds[z]
                        mss=msss[a]
                        start_year=1932
                        train_years=int(len(processed_train_data)*subsample)
                        validation_years=len(processed_train_data)-train_years               
                        result=rf_model_training(processed_train_data,target_1,start_year,train_years,validation_years,subsample,ne,md,mss,X_test) 
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
        model_file = 'RandomForest'+'-'+city+'-'+feature_name+'.joblib'
        model_path = os.path.join(folder_path, model_file)
        joblib.dump(fmodel,model_path )
        print(combination)
        
    subsample=0.7
    print("subsample:",subsample)
    start_year=1932
    train_years=int(len(processed_train_data)*subsample)
    validation_years=len(processed_train_data)-train_years
    model_file = 'RandomForest'+'-'+city+'-'+feature_name+'.joblib'
    model_path = os.path.join(folder_path, model_file)
    model = joblib.load(model_path)
    result = rf_model_testing(model,processed_train_data,target_1,start_year,train_years,validation_years,subsample,X_test)
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
    plt.title("Randomforest Model")
    plt.show()
    print('train model totally using ',used_train,'pieces of data','mse:',MSE)
    return plt,va_pred,years

'''
source_data = pd.read_csv('data.csv')
city = 'USA'
feature_name = 'oil_price'
lightgbm_func(source_data,city, feature_name,train = 1)
'''