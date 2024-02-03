#import packages
import pandas as pd
import numpy as np
from xgboost import XGBRegressor as XGBR
import matplotlib.pyplot as plt

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

def xgb_model_training(data,target,start_year,train_years,validation_years,test_years,subsample,learning_rate,max_depth):
    train_start=pd.to_datetime(str(start_year))
    train_end=train_start+pd.DateOffset(years=train_years)-pd.DateOffset(days=1)
    validation_start=train_end+pd.DateOffset(days=1)
    validation_end=validation_start+pd.DateOffset(years=validation_years)-pd.DateOffset(days=1)
    test_start=train_end+pd.DateOffset(days=1)
    test_end=test_start+pd.DateOffset(years=test_years)-pd.DateOffset(days=1)
    
    feature=data.columns.tolist()
    X_train=data.loc[train_start:train_end][feature].values
    y_train=target.loc[train_start:train_end].values
    X_validation=data.loc[validation_start:validation_end][feature].values
    y_validation=target.loc[validation_start:validation_end].values
    
    print("start model training:")
    xgb_reg=XGBR(subsample=subsample,learning_rate=learning_rate,max_depth=max_depth)
    xgb_reg.fit(X_train,y_train)
    y_pred=xgb_reg.predict(X_validation)
    
    #calculate mse of validation set
    va_y=y_validation.reshape(-1,1)
    va_y_pred=y_pred.reshape(-1,1)
    mean_square_error=np.mean((va_y-va_y_pred)**2)
    return mean_square_error,va_y,va_y_pred

#Set year as index
def xgboost_func(source_data,city, feature_name):
    source_data['year'] = pd.to_datetime(source_data['year'], format='%Y')
    source_data = source_data.sort_values(by=['year'])
    source_data = source_data.set_index(['year',])
    train_data=source_data[source_data['cty_name']==city]

    #target列设置为oil_price_2000
    target_1=pd.DataFrame(train_data[feature_name],index=train_data[feature_name].index,columns=[feature_name])
    target_1

    #删去target列,cty_name和id
    train_data = train_data.drop(feature_name, axis=1)
    processed_train_data=processing_data(train_data)
    processed_train_data=zscore_data(processed_train_data)
    processed_train_data=processed_train_data.iloc[:,2:]
    processed_train_data

    subsample_list=[0.7,0.8,0.85]
    learning_rate_list=[0.01,0.1,0.2]
    max_depth_list=[3,5,10,15]
    MSE=1000000
    combination=[0,0,0]
    for x in range(0,2):
        for y in range(0,3):
            for z in range(0,3):
                subsample=subsample_list[x]
                learning_rate=learning_rate_list[y]
                max_depth=max_depth_list[z]
                print("subsample:",subsample)
                print("learning_rate:",learning_rate)
                print("max_depth:",max_depth)
                start_year=1932
                train_years=int(len(processed_train_data)*subsample)
                validation_years=len(processed_train_data)-train_years
                test_years = 0
                
                result=xgb_model_training(processed_train_data,target_1,start_year,train_years,validation_years,test_years,subsample,learning_rate,max_depth)
                print(result)

                #find best prediction:
                if result[0]<MSE:
                    MSE = result[0]
                    va_y = result[1]
                    va_pred = result[2]
                    used_train = train_years
                    used_data = train_years + validation_years
                    combination[0] = x
                    combination[1] = y
                    combination[2] = z
                    
    #vasualization:
    x = range(0,len(va_pred))
    years = []
    start_year = 1932
    for i in x:
        years.append(pd.to_datetime(str(start_year+used_train))+pd.DateOffset(years=i))
    #fig,ax = plt.subplots(figsize = (10,6))
    plt.plot(years,va_pred,label = 'model test')
    x = range(1,len(va_pred)+1)
    plt.plot(years,va_y,label = 'real value')
    plt.xlabel('last predicted years')
    plt.xlabel('last predicted years')
    plt.legend()
    plt.show()
    print('train model totally using ',used_train,'pieces of data','mse:',MSE)
    print('best combination is: subsample_rate:',combination[0],'learning_rate_list:',combination[1],'max_depth_list:',combination[2])
    return plt