import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import statsmodels.api as sm
#import plotly.graph_objects as go


# 画各个城市时间函数的趋势线图
def trend(data, cities, feature_name,start,tail):
    num = len(cities)
    data_time = data[((start<=data['year']) & (data['year']<=tail))]
    fig,ax = plt.subplots(figsize = (10,6))
    if num > 6:
        data_group = data_time.groupby('year').mean()
        ax.plot(data_group[feature_name],label='Averaged curve')
    else:
        for i in range(num):
            filtered_data = data_time[data_time['country']==cities[i]]
            ax.plot(filtered_data['year'],filtered_data[feature_name],label=cities[i])
    
    ax.set_xlabel('Time')
    ax.set_ylabel(feature_name)
    ax.set_title(feature_name + ' of different countries')
    ax.legend()
    return fig


#某个国家不同特征的时间趋势比较
def corr_features_cities(data, city, feature_names, start,tail):
    num = len(feature_names)
    filtered_data = data[(start<=data['year']) & (data['year']<=tail) &(data['country'] == city)]
    fig,ax = plt.subplots(figsize = (10,6))   
    for i in range(num):
        ax.plot(filtered_data['year'],filtered_data[feature_names[i]],label=feature_names[i])
    ax.legend()
    ax.set_xlabel('time')
    ax.set_ylabel('value')
    ax.set_title('Features of '+city)
    return fig
        
        
# 不同城市平均指标的对比(问题：不同国家不同指标的绝对值差别较大)
def corr_cities(data, cities, feature_names,start,tail):
    num_cities,num_features = len(cities), len(feature_names)
    data_time = data[((start<=data['year']) & (data['year']<=tail))]
    data_time = data_time[['country'] + feature_names]
    fig,ax = plt.subplots(figsize = (10,6))
    
    data_group = data_time[data_time['country'].map(lambda x: x in cities)].groupby('country').mean()
    # # 筛选出城市在指定城市列表中的数据
    # data_cities = data_time[data_time['country'].isin(cities)]
    # # 分组并计算均值
    # data_group = data_cities.groupby('country')[feature_names].mean()
    x = range(num_cities)
    bar_width = 0.15
    for i in range(num_features):
        ax.bar([j+i*bar_width for j in x],data_group[feature_names[i]].values,align='center',label=feature_names[i],width=bar_width)
    
    ax.set_xticks([j + num_features*bar_width/2 for j in x],cities)
    ax.set_xlabel('country')
    ax.set_ylabel('value')
    ax.set_title('comparison between countries')
    ax.legend(loc='upper left')        
    return fig
# def corr_cities(data, cities, feature_names, start, tail):
#     fig = go.Figure()  # 创建一个空的Plotly图表对象
    
#     for feature in feature_names:
#         for city in cities:
#             # 筛选指定年份、城市和特征的数据
#             filtered_data = data[(data['year'] >= start) & (data['year'] <= tail) & (data['country'] == city)]
            
#             # 计算平均指标
#             avg_data = filtered_data[feature].mean()
            
#             # 添加城市和特征的平均指标到图表中
#             fig.add_trace(go.Bar(x=[f"{city}-{feature}"], y=[avg_data], name=f"{city}-{feature}"))
    
#     # 设置图表布局和标签
#     fig.update_layout(
#         xaxis_title='city-feature',
#         yaxis_title='averaged_feature ',
#         title='Comparison of average indicators for different features in different cities'
#     )
#     return fig


# 看石油和天然气之间的关系(相关系数热力图)
def corr_features(data, feature_names,start, tail):

    data_time = data[(start<=data['year']) & (data['year']<=tail)]
    fig,ax = plt.subplots(figsize = (10,6))
    
    matrix = np.array(data_time[feature_names].corr())
    im = ax.imshow(matrix, cmap='hot', interpolation='nearest')
    fig.colorbar(im,ax=ax) 
    # 标出矩阵的值
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, round(matrix[i, j], 2), ha='center', va='center', color='green')

    # 添加栅格的边框
    ax.set_xticks(np.arange(-.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, matrix.shape[0], 1), minor=True)
    plt.grid(which='minor', color='black', linestyle='-', linewidth=2)
    ax.set_xticks(range(matrix.shape[1]), [str(feature_names[i]) for i in range(matrix.shape[1])])
    ax.set_yticks(range(matrix.shape[0]), [str(feature_names[i]) for i in range(matrix.shape[0])])
    ax.set_title('correlation matrix')
    return fig
    
# 季节趋势分解某个国家某个特征
# 某一个国家，某一个特征的趋势分解(要求：不可以有缺失值，时间要连续)
#定义时序数列转换为数据集的函数
def season(data, city, feature_name,start,tail):
    filtered_data = data[(start<=data['year']) & (data['year']<=tail) & 
                     (data['country'] == city)]
    filtered_data = filtered_data[['year']+[feature_name]]
    
    # Construct the seasonal decomposition model
    decomposition = sm.tsa.seasonal_decompose(filtered_data[feature_name], model='additive', period=12)
        
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(filtered_data['year'],filtered_data[feature_name], label='Original')
    ax.plot(filtered_data['year'],decomposition.trend, label='Trend')
    ax.plot(filtered_data['year'],decomposition.seasonal, label='Seasonal')
    ax.plot(filtered_data['year'],decomposition.resid, label='Residual')
    ax.set_title('Result of season decomposition model')
    ax.legend(loc='upper left')    
    ax.set_xlabel('time')
    ax.set_ylabel('value')
    return fig



if __name__ == "__main__":
    # Read data
    data = pd.read_csv('../dataset/data.csv')
    data.head()
    #fig = season(data, 'Afghanistan', 'oil_price',1978,2014)
    #plt.show()
    fig = trend(data,['Afghanistan','Albania'],'gas_product',2000,2010)
    plt.show()
    fig = corr_features_cities(data,'Afghanistan',['oil_price','oil_exports'],2000,2010)
    plt.show()

    fig = corr_cities(data,['Algeria','Argentina'],['gas_product','oil_exports','gas_price'],2000,2010)
    plt.show()
    fig = corr_features(data,['gas_product','oil_exports','gas_price'],1950,2010)
    plt.show()











