import lstm as lstm
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
data = pd.read_csv('./train/data.csv')

# 国家特征组合的筛选
def select_pred_county_feature(data):
    '''
    输出：一个list，list的每个元素是个tuple，(country, feature)
    '''
    country_lst = data['country'].unique().tolist()
    feature_lst = data.columns[3:].tolist()
    res = []
    for country in country_lst:
        for feature in feature_lst:
            filtered_data = data[feature][data['country'] == country]
            if sum(filtered_data.isnull()) < len(filtered_data)*0.2 and sum(abs(filtered_data) <= 1e-5) < len(data)*0.1: #缺失值和0值少于20%
                res.append((country,feature))
    return res

tuple_process = select_pred_county_feature(data)
with open('./train/tuple_correct.pkl', 'rb') as f:
    tuple_correct = pickle.load(f)
dic = {}
for t in tuple_correct:
    if t[0] in dic.keys():
        dic[t[0]].append(t[1])
    else:
        dic[t[0]] = [t[1]]
with open('./train/dict_correct.pkl', 'wb') as f:
    pickle.dump(dic, f)
