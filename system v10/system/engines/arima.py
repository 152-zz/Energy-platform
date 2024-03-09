import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

def pred_arima(data, city, feature_name, start, tail):
    filtered_data = data[(start<=data['year']) & (data['year']<=tail) & 
                     (data['country'] == city)].set_index('year')
    filtered_data = filtered_data[feature_name]
    fig_vis_data, ax = plt.subplots(figsize=(10, 8))
    # 可视化时间序列数据
    ax.plot(filtered_data)
    ax.set_title('Time Series Data')

    # 绘制自相关函数（ACF）和偏自相关函数（PACF）图
    fig_acf = plot_acf(filtered_data, lags=20)
    fig_pacf = plot_pacf(filtered_data, lags=20)

    # 拟合 ARIMA 模型
    model = ARIMA(filtered_data, order=(1, 0, 1))  # 这里使用 ARIMA(p, d, q) 中 p=1, d=0, q=1 的参数
    model_fit = model.fit()

    # 打印模型的摘要信息

    # 进行预测
    n_forecast = 10  # 预测未来 10 个时间点的值
    forecast = model_fit.get_forecast(steps=n_forecast)
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int()

    # 可视化预测结果
    fig_pred, ax = plt.subplots(figsize=(10, 8))
    fitted_values = model_fit.fittedvalues
    n = filtered_data.index[-1]
    ax.plot(filtered_data, label='Actual')
    ax.plot(np.arange(n, n+n_forecast), forecast_mean, label='Forecast')
    ax.plot(fitted_values,label = 'Fit')
    plt.fill_between(np.arange(n, n+n_forecast), forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='gray', alpha=0.2)
    ax.legend()
    return (fig_vis_data, fig_acf, fig_pacf, fig_pred), model_fit.summary()


# if __name__ == "__main__":
#     # Read data
#     data = pd.read_csv('./dataset/Oil and Gas 1932-2014.csv')
#     figs,result = pred_arima(data, 'Afghanistan', 'oil_price_2000', 1934, 2014)
    
    