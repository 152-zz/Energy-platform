import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 画各个城市时间函数的趋势线图
def trend(data, cities, feature_name,start,tail):
    num = len(cities)
    data_time = data[((start<=data['year']) & (data['year']<=tail))]
    #fig,ax = plt.subplots(figsize = (10,6))
    lines = []
    if num > 6:
        data_group = data_time.groupby('year').mean()
        lines.append({'x':np.array(data_group['year']),'y':np.array(data_group[feature_name]),'label':'Averaged curve'})
        #ax.plot(data_group[feature_name],label='Averaged curve')
    else:
        for i in range(num):
            filtered_data = data_time[data_time['country']==cities[i]]
            #print(type(filtered_data))
            lines.append({'x':np.array(filtered_data['year']),'y':np.array(filtered_data[feature_name]),'label':cities[i]})
            #ax.plot(filtered_data['year'],filtered_data[feature_name],label=cities[i])
    
    #ax.set_xlabel('Time')
    #ax.set_ylabel(feature_name)
    #ax.set_title(feature_name + ' of different countries')
    #ax.legend()
    return lines,

#某个国家不同特征的时间趋势比较
def corr_features_cities(data, city, feature_names, start,tail):
    num = len(feature_names)
    filtered_data = data[(start<=data['year']) & (data['year']<=tail) &(data['country'] == city)]
    lines = []
    #fig,ax = plt.subplots(figsize = (10,6))   
    for i in range(num):
        lines.append({'x':np.array(filtered_data['year']),'y':np.array(filtered_data[feature_names[i]]),'label':feature_names[i]})
        #ax.plot(filtered_data['year'],filtered_data[feature_names[i]],label=feature_names[i])
    #ax.legend()
    #ax.set_xlabel('time')
    #ax.set_ylabel('value')
    #ax.set_title('Features of '+city)
    return lines,

# 不同城市平均指标的对比(问题：不同国家不同指标的绝对值差别较大)
def corr_cities(data, cities, feature_names,start,tail):
    num_cities,num_features = len(cities), len(feature_names)
    data_time = data[((start<=data['year']) & (data['year']<=tail))]
    data_time = data_time[['country'] + feature_names]
    bars = []
    #fig,ax = plt.subplots(figsize = (10,6))
    
    data_group = data_time[data_time['country'].map(lambda x: x in cities)].groupby('country').mean()
    # # 筛选出城市在指定城市列表中的数据
    # data_cities = data_time[data_time['country'].isin(cities)]
    # # 分组并计算均值
    # data_group = data_cities.groupby('country')[feature_names].mean()
    x = range(num_cities)
    bar_width = 0.15
    for i in range(num_features):
        bars.append({'x':np.array([j+i*bar_width for j in x]),
                     'y':np.array(data_group[feature_names[i]].values),
                    'label':feature_names[i],'align' : 'center','width':bar_width})
        #ax.bar([j+i*bar_width for j in x],data_group[feature_names[i]].values,align='center',label=feature_names[i],width=bar_width)
    graph_params = {'set_xticks':([j + num_features*bar_width/2 for j in x],cities),'bar_width':bar_width}
    
    #ax.set_xticks([j + num_features*bar_width/2 for j in x],cities)
    #ax.set_xlabel('country')
    #ax.set_ylabel('value')
    #ax.set_title('comparison between countries')
    #ax.legend(loc='upper left')        
    return bars,graph_params
# 看石油和天然气之间的关系(相关系数热力图)
def corr_features(data, feature_names,start, tail):

    data_time = data[(start<=data['year']) & (data['year']<=tail)]
    #fig,ax = plt.subplots(figsize = (10,6))
    
    matrix = np.array(data_time[feature_names].corr())
    '''
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
    '''
    return {'x':matrix},


if __name__ == "__main__":
    # Read data
    data = pd.read_csv('../dataset/data.csv')
    data.head()
    #fig = season(data, 'Afghanistan', 'oil_price',1978,2014)
    #plt.show()
    lines, = trend(data,['Afghanistan','Albania'],'gas_product',2000,2010)
    lines, = corr_features_cities(data,'Afghanistan',['oil_price','oil_exports'],2000,2010)
    bars,graph_params = corr_cities(data,['Algeria','Argentina'],['gas_product','oil_exports','gas_price'],2000,2010)
    grid_plot = corr_features(data,['gas_product','oil_exports','gas_price'],1950,2010)
