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
import engines.xgboost_sample as xgb
import os

# 定义页面标识
page = st.sidebar.selectbox('Choose your page', ['Main Page',  'Visualization','Analysis', 'Prediction'])

current_dir = os.path.dirname(__file__)
relative_path = os.path.join(current_dir, 'dataset/data.csv')
data = pd.read_csv(relative_path)



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
    
    oil_featrues = ['oil_prod32_14','oil_price_2000','oil_value_2000','oil_exports',
                    'oil_pro_per','oil_val_per']
    gas_features = ['gas_prod55_14','gas_price_2000','gas_value_2000','gas_exports',
                    'gas_pro_per','gas_val_per']
    
    features = []
    if energy_option == 'Oil':
        features = oil_featrues
    elif energy_option == 'Gas':
        features = gas_features
        
    cities = data["cty_name"].unique().tolist()
    
    st.sidebar.text('2.year')
    start_year = st.sidebar.text_input('start year','1932')
    end_year = st.sidebar.text_input('end year','2014')
    
    if start_year > end_year:
        st.sidebar.error("The end year must later than the start year!")
    else:
        patterns = ['mutiple cities with one feature','mutiple features in one city']
        pattern_option = st.selectbox('please select a pattern',patterns)
        
        if pattern_option == 'mutiple cities with one feature':
            cities_option = st.multiselect('please select one or more cities',cities)
            feature_option = st.selectbox('please select one feature',features)
            
            fig = ana.trend(data,cities_option,feature_option,int(start_year),int(end_year))
            if cities_option:
                st.pyplot(fig)
                
        elif pattern_option == 'mutiple features in one city':
            city_option = st.selectbox('please select one or more cities',cities)
            features_option = st.multiselect('please select one feature',features)
            fig = ana.corr_features_cities(data,city_option,features_option,int(start_year),int(end_year))
            if features_option:
                st.pyplot(fig)



elif page == 'Analysis':
    col1, col2, col3 = st.columns((1, 4, 1))
    with col2:
        st.markdown("# DATA ANLYSIS")
    with col3:
        st.markdown("[![GitHub](https://img.shields.io/badge/-GitHub-black?logo=github&style=flat-square)](https://github.com/msdm-ust/energyintel_data_platform)", unsafe_allow_html=True)
    st.sidebar.write('Sidebar for Analysis')
    
    # 添加用于选择是否检测和剔除outliers的选项
    def outlier_detect(data,city,feature):
        filtered_data = data[(data['cty_name'] == city)]
        fig,ax = plt.subplots(figsize = (10,6))
        # 绘制箱线图
        ax.boxplot(filtered_data[feature])
        return fig


    outlier_option = st.sidebar.radio('Outlier Options', ['Detect', 'Drop'])
    if outlier_option == 'Detect':
        st.title("Detecting outliers...")
        cities_option = st.selectbox("please select one or more cities", data['cty_name'].unique())
        features_option = st.selectbox("choose one feature", [col for col in data.columns if col not in ['cty_name', 'year',"id"]])
        fig = outlier_detect(data,cities_option,features_option)
        if features_option:
            st.pyplot(fig)



    # 这里添加检测outliers的代码
    elif outlier_option == 'Drop':
        data = data.bfill()
        st.write('Outliers have been dropped...')
        # 这里添加剔除outliers的代码
    


        #patterns = ['correlation with cities','correlation with features',"seasonal trend decomposition"]
        patterns = ['correlation with cities','correlation with features']
        pattern_option = st.selectbox('please select a pattern',patterns)
        

        
        if pattern_option == 'correlation with cities':
            st.title("correlation with cities")
            # 用户输入
            cities_option = st.multiselect("please select one or more cities", data['cty_name'].unique(), key="cities")

            # 确保年份选择逻辑正确
            min_year, max_year = int(data['year'].min()), int(data['year'].max())
            start_year = st.slider("choose start year", min_year, max_year, min_year)
            end_year = st.slider("choose end year", min_year, max_year, max_year)

            # 确保用户不能选择结束年份小于开始年份
            if end_year < start_year:
                st.error("The end year must later than the start year!")
            else:
                feature_option = st.multiselect("choose one or more features", [col for col in data.columns if col not in ['cty_name', 'year']],key="feature_names")

            # 如果用户已经做出选择，则显示图表
            if cities_option and feature_option:
                fig = ana.corr_cities(data, cities_option, feature_option, start_year, end_year)
                st.pyplot(fig)
                
        elif pattern_option == 'correlation with features':
            st.title("correlation with features")

            # 确保年份选择逻辑正确
            min_year, max_year = int(data['year'].min()), int(data['year'].max())
            start_year = st.slider("choose start year", min_year, max_year, min_year)
            end_year = st.slider("choose end year", min_year, max_year, max_year)

            # 确保用户不能选择结束年份小于开始年份
            if end_year < start_year:
                st.error("The end year must later than the start year!")
            else:
                features_option = st.multiselect("choose one or more features", [col for col in data.columns if col not in ['cty_name', 'year']])


            fig = ana.corr_features(data,features_option,start_year,end_year)
            if features_option:
                st.pyplot(fig)

        else:
            st.title("Seasonal Trend Decomposition")
            # 用户输入
            cities_option = st.selectbox("please select one city", data['cty_name'].unique(), key="city_season")

            # 确保年份选择逻辑正确
            min_year, max_year = int(data['year'].min()), int(data['year'].max())
            start_year = st.slider("choose start year", min_year, max_year, min_year)
            end_year = st.slider("choose end year", min_year, max_year, max_year)

            # 确保用户不能选择结束年份小于开始年份
            if end_year < start_year:
                st.error("The end year must later than the start year!")
            else:
                feature_option = st.selectbox("choose one feature", [col for col in data.columns if col not in ['cty_name', 'year']],key="feature_names")

            # 如果用户已经做出选择，则显示图表
            if cities_option and feature_option:
                fig = ana.season(data, cities_option, feature_option, start_year, end_year)
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

    oil_featrues = ['oil_prod32_14','oil_price_2000','oil_value_2000','oil_exports',
                    'oil_pro_per','oil_val_per']
    gas_features = ['gas_prod55_14','gas_price_2000','gas_value_2000','gas_exports',
                    'gas_pro_per','gas_val_per']

    features = []
    if energy_option == 'Oil':
        features = oil_featrues
    elif energy_option == 'Gas':
        features = gas_features

    
    cities = data["cty_name"].unique().tolist()
    default_cities_index = cities.index('United States')
    default_features_index = features.index('oil_price_2000')

    model_option = st.sidebar.radio('2.Model Options', ['Linear', 'Tree','Random Forest','LSTM','XGBoost'])

    city_option = st.selectbox('please select one cities',cities,index = default_cities_index)
    feature_option = st.selectbox('please select one feature',features,index = default_features_index)

    model_option = st.sidebar.radio('2.Model Options', ['Linear', 'Tree','Random Forest',
                                                         'LSTM','XGBoost','Arima'])
    if model_option == 'Linear':
        st.write("Linear Model Result:")
    elif model_option == 'Tree':
        st.write("Tree Model Result:")
    elif model_option == 'Random Forest':
        st.write("Random Forest Model Result:")
    elif model_option == 'LSTM':
        st.write("LSTM Model Result:")
    elif model_option == 'XGBoost':
        st.write("XGBoost Model Result:")
        fig = xgb.xgboost_func(data,city_option,feature_option)
        st.pyplot(fig)
    elif model_option == 'Arima':
        st.title("ARIMA")
        # 用户输入
        cities_option = st.selectbox("please select one or more cities", data['cty_name'].unique(), key="cities")

        # 确保年份选择逻辑正确
        min_year, max_year = int(data['year'].min()), int(data['year'].max())
        start_year = st.slider("choose start year", min_year, max_year, min_year)
        end_year = st.slider("choose end year", min_year, max_year, max_year)

        # 确保用户不能选择结束年份小于开始年份
        if end_year < start_year:
            st.error("结束年份不能小于开始年份")
        else:
            feature_option = st.selectbox("choose one or more features", [col for col in data.columns if col not in ['cty_name', 'year',"id"]],key="feature_names")

        # 如果用户已经做出选择，则显示图表
        if cities_option and feature_option:
            figs,result =ari.pred_arima(data, cities_option, feature_option, start_year, end_year)
            st.pyplot(figs[0])
            st.pyplot(figs[1])
            st.pyplot(figs[2])
            st.pyplot(figs[3])
            st.write(result)

