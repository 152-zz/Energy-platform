"""
Created on Tue Jan 23 10:25:46 2024

@author: lhy
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import engines.analysis as ana
import engines.arima as ari
import engines.xgboost as xgb
import engines.lstm as lstm
import engines.GRU as gru
from engines.GRU import RNNModel
import engines.Randomforest as rf
import engines.LightGBM as lg
import os

# 定义页面标识
page = st.sidebar.selectbox('Choose your page', ['Main Page', 'Visualization','Analysis', 'Prediction'])

current_dir = os.path.dirname(__file__)
relative_path = os.path.join(current_dir, 'dataset/data.csv')
data = pd.read_csv(relative_path)

# A sample feature mapping

feature_map_total = {v:"".join([f"{i.capitalize()} " for i in v.split("_")])[0:-1]
               for v in [col for col in data.columns if col not in ['country', 'year',"id"]]}
feature_revise_map_total = {v:k for k,v in feature_map_total.items()}
feature_map_oil = {v:"".join([f"{i.capitalize()} " for i in v.split("_")])[0:-1]
               for v in [col for col in data.columns if col in ['oil_product',
               'oil_price','oil_value','oil_exports','oil_pro_person','oil_val_person']]}
feature_revise_map_oil = {v:k for k,v in feature_map_oil.items()}
feature_map_gas = {v:"".join([f"{i.capitalize()} " for i in v.split("_")])[0:-1]
               for v in [col for col in data.columns if col in ['gas_product',
               'gas_price','gas_value','gas_exports','gas_pro_person','gas_val_person']]}
feature_revise_map_gas = {v:k for k,v in feature_map_gas.items()}

oil_features = ['oil_product','oil_price','oil_value','oil_exports',
                'oil_pro_person','oil_val_person']
gas_features = ['gas_product','gas_price','gas_value','gas_exports',
                'gas_pro_person','gas_val_person']

if page == 'Main Page':
    # Logo and Navigation
    col1, col2, col3 = st.columns((1, 4, 1))
    with col2:
        st.markdown("# ENERGY PLATFORM")
    with col3:
        st.markdown("[![GitHub](https://img.shields.io/badge/-GitHub-black?logo=github&style=flat-square)](https://github.com/msdm-ust/energyintel_data_platform)", unsafe_allow_html=True)
    st.sidebar.write('Sidebar for Main Page')
    st.write('Content for Main Page')
    image_path = 'system/image1.jpeg'
    st.image(image_path, caption='Oil & Gas', use_column_width=True)

elif page == 'Visualization':
    # Logo and Navigation
    col1, col2, col3 = st.columns((1, 4, 1))
    with col2:
        st.markdown("# Visualization")
    with col3:
        st.markdown("[![GitHub](https://img.shields.io/badge/-GitHub-black?logo=github&style=flat-square)](https://github.com/msdm-ust/energyintel_data_platform)", unsafe_allow_html=True)
    st.sidebar.write('Sidebar for Visualization')
    energy_option = st.sidebar.radio('1.Energy Options', ['Oil', 'Gas'])
    
    if energy_option == 'Oil':
        feature_map = feature_map_oil
        features = oil_features
        feature_revise_map = feature_revise_map_oil
    elif energy_option == 'Gas':
        feature_map = feature_map_gas
        features = gas_features
        feature_revise_map = feature_revise_map_gas
        
    cities = data["country"].unique().tolist()
    
    min_year, max_year = int(data['year'].min()), int(data['year'].max())
    start_year = st.slider("Choose start year", min_year, max_year, min_year)
    end_year = st.slider("Choose end year", min_year, max_year, max_year)
    if end_year < start_year:
            st.error("The end year must later than the start year!")
    else:
        patterns = ['Mutiple countries with one feature','Mutiple features in one country']
        pattern_option = st.selectbox('Please select a pattern',patterns)
        
        if pattern_option == 'Mutiple countries with one feature':
            cities_option = st.multiselect('Please select one or more countries',cities)
            feature_option = st.selectbox('Please select one feature',[feature_map[col] for col in data.columns if col in features])
            feature_option = feature_revise_map[feature_option]
            lines, = ana.trend(data,cities_option,feature_option,int(start_year),int(end_year))
            if cities_option:
                df = pd.DataFrame(
                        np.array([line['y'] for line in lines]).T,
                        columns = cities_option,
                        index = lines[0]['x'] )
                st.title(feature_map[feature_option]+ ' of Different Countries')   
                st.line_chart(df)
                
        elif pattern_option == 'Mutiple features in one country':
            city_option = st.selectbox('Please select one country',cities)
            features_option = st.multiselect('Please select one or more features',[feature_map[col] for col in data.columns if col in features])
            feature_option = [feature_revise_map[feature] for feature in features_option]
            lines, = ana.corr_features_cities(data,city_option,feature_option,int(start_year),int(end_year))
            if features_option:
                df = pd.DataFrame(
                        np.array([line['y'] for line in lines]).T,
                        columns = features_option,
                        index = lines[0]['x'] )
                st.title('Different Features of '+ city_option)   
                st.line_chart(df)

        



elif page == 'Analysis':
    col1, col2, col3 = st.columns((1, 4, 1))
    with col2:
        st.markdown("# DATA ANLYSIS")
    with col3:
        st.markdown("[![GitHub](https://img.shields.io/badge/-GitHub-black?logo=github&style=flat-square)](https://github.com/msdm-ust/energyintel_data_platform)", unsafe_allow_html=True)
    st.sidebar.write('Sidebar for Analysis')
    
    # 添加用于选择是否检测和剔除outliers的选项
    def outlier_detect(data,city,feature):
        filtered_data = data[(data['country'] == city)]
        fig,ax = plt.subplots(figsize = (10,6))
        # 绘制箱线图
        ax.boxplot(filtered_data[feature])
        ax.set_xlabel("feature")
        ax.set_ylabel("value")
        ax.set_title("Box Plot")
        return fig

    outlier_option = st.sidebar.radio('Outlier Options', ['Detect', 'Drop'])
    if outlier_option == 'Detect':
        st.title("Detecting outliers...")
        cities_option = st.selectbox("Please select one or more countries", data['country'].unique())
        features_option = st.selectbox("Choose one feature", [feature_map_total[col] for col in data.columns if col not in ['country', 'year',"id"]])
        fig = outlier_detect(data,cities_option,feature_revise_map_total[features_option])
        if features_option:
            st.pyplot(fig)

    # 这里添加检测outliers的代码
    elif outlier_option == 'Drop':
        data = data.bfill()
        st.write('Outliers have been dropped...')
        # 这里添加剔除outliers的代码

        #patterns = ['correlation with cities','correlation with features',"seasonal trend decomposition"]
        patterns = ['Correlation with countries','Correlation with features']
        pattern_option = st.selectbox('Please select a pattern',patterns)
        
        if pattern_option == 'Correlation with countries':
            st.title("Correlation with countries")
            # 用户输入
            cities_option = st.multiselect("Please select one or more countries", data['country'].unique(), key="cities")

            # 确保年份选择逻辑正确
            min_year, max_year = int(data['year'].min()), int(data['year'].max())
            start_year = st.slider("Choose start year", min_year, max_year, min_year)
            end_year = st.slider("Choose end year", min_year, max_year, max_year)

            # 确保用户不能选择结束年份小于开始年份
            if end_year < start_year:
                st.error("The end year must later than the start year!")
            else:
                feature_option = st.multiselect("Choose one or more features", [feature_map_total[col] for col in data.columns if col not in ['country', 'year','id','population']],key="feature_names")

            # 如果用户已经做出选择，则显示图表
            if cities_option and feature_option:
                bars,graph_params = ana.corr_cities(data, cities_option, [feature_revise_map_total[feature] for feature in feature_option], start_year, end_year)
                df = pd.DataFrame(
                           np.array([bar['y'] for bar in bars]).T,
                           columns = [bar['label'] for bar in bars],
                           index = graph_params["set_xticks"][1])
                st.title('correlation with cities')   
                st.bar_chart(df)
                
        elif pattern_option == 'Correlation with features':
            st.title("Correlation with features")

            # 确保年份选择逻辑正确
            min_year, max_year = int(data['year'].min()), int(data['year'].max())
            start_year = st.slider("Choose start year", min_year, max_year, min_year)
            end_year = st.slider("Choose end year", min_year, max_year, max_year)

            # 确保用户不能选择结束年份小于开始年份
            if end_year < start_year:
                st.error("The end year must later than the start year!")
            else:
                features_option = st.multiselect("Choose one or more features", [col for col in data.columns if col not in ['country', 'year','id']])


            fig = ana.corr_features(data,features_option,start_year,end_year)
            if features_option:
                st.pyplot(fig)

elif page == 'Prediction':
    # Logo and Navigation
    col1, col2, col3 = st.columns((1, 4, 1))
    with col2:
        st.markdown("# Prediction")
    with col3:
        st.markdown("[![GitHub](https://img.shields.io/badge/-GitHub-black?logo=github&style=flat-square)](https://github.com/msdm-ust/energyintel_data_platform)", unsafe_allow_html=True)

    energy_option = st.sidebar.radio('1.Energy Options', ['Oil', 'Gas'])
    if energy_option == 'Oil':
        st.write("Oil Prediction")
    elif energy_option == 'Gas':
        st.write("Gas Prediction")

    if energy_option == 'Oil':
        feature_map = feature_map_oil
        features = oil_features
        feature_revise_map = feature_revise_map_oil
    elif energy_option == 'Gas':
        feature_map = feature_map_gas
        features = gas_features
        feature_revise_map = feature_revise_map_gas
    
    cities = data["country"].unique().tolist()
    default_cities_index = cities.index('United States')
    if energy_option == 'Oil':
        default_features_index = features.index('oil_price')
    elif energy_option == 'Gas':
        default_features_index = features.index('gas_price')

    model_option = st.sidebar.radio('2.Model Options', ['Linear','Decision Tree','Random Forest','LightGBM','RNN','GRU','LSTM','XGBoost','Arima'])

    city_option = st.selectbox('Please select one country',cities,index = default_cities_index)
    feature_option = st.selectbox('Please select one feature',[feature_map[col] for col in data.columns if col in features],index = default_features_index)
    feature_option = feature_revise_map[feature_option]
    if model_option == 'Linear':
        st.write("Linear Model Result:")
    elif model_option == 'Decision Tree':
        st.write("Decision Tree Model Result:")
    elif model_option == 'Random Forest':
        st.write("Random Forest Model Result:")
        plt,pred,years = rf.randomforest_func(data,city_option,feature_option,0)
        data = {}
        df = pd.DataFrame(data)
        for i in range(len(years) - 5,len(years)):
            y = years[i].strftime("%Y-%m-%d")
            y = y[:4]
            df[y] = pred[i]
        st.write(df)
        st.pyplot(plt)
    elif model_option == 'LightGBM':
        st.write('LightGBM Model Result')
        plt,pred,years = lg.lightgbm_func(data,city_option,feature_option,0)
        data = {}
        df = pd.DataFrame(data)
        for i in range(len(years) - 5,len(years)):
            y = years[i].strftime("%Y-%m-%d")
            y = y[:4]
            df[y] = pred[i]
        st.write(df)
        st.pyplot(plt)
    elif model_option == 'RNN':
        st.write("RNN Model Result:")
    elif model_option == 'GRU':
        st.write("GRU Model Result:")
        fig = gru.GRU_model("Afghanistan","oil_price",1934,2014)
        st.pyplot(fig)
    elif model_option == 'LSTM':
        st.write("LSTM Model Result:")
        fig,pred,years = lstm.predict(data,city_option,feature_option,1)
        data = {}
        df = pd.DataFrame(data)
        for i in range(len(years) - 5,len(years)):
            y = years[i].strftime("%Y-%m-%d")
            y = y[:4]
            df[y] = pred[i]
        st.write(df)
        st.pyplot(fig)
    elif model_option == 'XGBoost':
        st.write("XGBoost Model Result:")
        fig,pred,years = xgb.xgboost_func(data,city_option,feature_option,1)
        data = {}
        df = pd.DataFrame(data)
        for i in range(len(years) - 5,len(years)):
            y = years[i].strftime("%Y-%m-%d")
            y = y[:4]
            df[y] = pred[i]
        st.write(df)
        st.pyplot(fig)
    elif model_option == 'Arima':
        st.title("ARIMA")
        # 用户输入
        cities_option = st.selectbox("Please select one or more countries", data['country'].unique(), key="cities")

        # 确保年份选择逻辑正确
        min_year, max_year = int(data['year'].min()), int(data['year'].max())
        start_year = st.slider("Choose start year", min_year, max_year, min_year)
        end_year = st.slider("Choose end year", min_year, max_year, max_year)

        # 确保用户不能选择结束年份小于开始年份
        if end_year < start_year:
            st.error("The end year must later than the start year!")
        else:
            feature_option = st.selectbox("Choose one or more features", [col for col in data.columns if col not in ['country', 'year',"id"]],key="feature_names")

        # 如果用户已经做出选择，则显示图表
        if cities_option and feature_option:
            figs,result =ari.pred_arima(data, cities_option, feature_option, start_year, end_year)
            st.pyplot(figs[0])
            st.pyplot(figs[1])
            st.pyplot(figs[2])
            st.pyplot(figs[3])
            st.write(result)
