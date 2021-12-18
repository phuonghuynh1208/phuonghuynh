
# pip install pyqt5
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
import pandas as pd
import seaborn as sns
# import pandas_profiling as pp
from sklearn.ensemble import ExtraTreesRegressor 
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
# %matplotlib inline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import *
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,cross_val_score
# import pandas_profiling as pp
import pickle
from fbprophet import Prophet 
from fbprophet.plot import add_changepoints_to_plot
import plotly.figure_factory as ff
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
# from xgboost import XGBRegressor
from sklearn.metrics import *
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima

# Source Code
data = pd.read_csv("avocado.csv")

#--------------
# GUI

st.title("Data Science Project 1")
st.header("#Hass  Avocado Price Prediction")
#---------------------
#Gui
menu = ["Business Objective & Understading", "1.Average Price Prediction","2.Avocado Price by region - next years prediction"]
choice=st.sidebar.selectbox("Menu", menu)
if choice == "Business Objective & Understading":
    st.subheader("Business Objective")
    st.write("""
    * Bơ “Hass”, một công ty có trụ sở tại Mexico, chuyên sản xuất nhiều loại quả bơ được bán ở Mỹ. Họ đã rất thành công trong những năm gần đây và muốn mở rộng. Vì vậy, họ muốn xây dựng mô hình hợp lý để dự đoán giá trung bình của bơ “Hass” ở Mỹ nhằm xem xét việc mở rộng các loại trang trại Bơ đang có cho việc trồng bơ ở các vùng khác..""")
    st.subheader("Business Understading")
    st.write("""* Hiện tại: Công ty kinh doanh quả bơ ở rất nhiều vùng của nước Mỹ với 2 loại bơ là bơ thường và bơ hữu cơ, được đóng gói theo nhiều quy chuẩn (Small/Large/XLarge Bags), và có 3 PLU (Product Look Up) khác nhau (4046, 4225, 4770). Công ty cần mô hình dự đoán giá để ra quyết định mở rộng kinh doanh""")
    st.image("Hass_avocado_2.jpg")
    st.write("""* => Mục tiêu/ Vấn đề: Xây dựng mô hình dự đoán giá trung bình của bơ “Hass” ở Mỹ => xem xét việc mở rộng sản xuất, kinh doanh.""")
    st.subheader("Data Understading")
    st.image("data.jpg")
    st.write("""Dữ liệu được lấy trực tiếp từ máy tính tiền của các nhà bán

* Dữ liệu được lấy trực tiếp từ máy tính tiền của các nhà bán lẻ dựa trên doanh số bán lẻ thực tế của bơ Hass.
* Dữ liệu đại diện cho dữ liệu lấy từ máy quét bán lẻ hàng tuần cho lượng bán lẻ (National retail volume- units) và giá bơ từ tháng 4/2015 đến tháng 3/2018.
* Giá Trung bình (Average Price) trong bảng phản ánh giá trên một đơn vị (mỗi quả bơ), ngay cả khi nhiều đơn vị (bơ) được bán trong bao.
* Mã tra cứu sản phẩm - Product Lookup codes (PLU’s) trong bảng chỉ dành cho bơ Hass, không dành cho các sản phẩm khác.""")
    st.write("""Dữ liệu dùng cho dự báo. Với các cột:
.""")
    st.write("""* Date - ngày ghi nhận
* AveragePrice – giá trung bình của một quả bơ
* Type - conventional / organic – loại: thông thường/ hữu cơ
* Region – vùng được bán
* Total Volume – tổng số bơ đã bán
* 4046 – tổng số bơ có mã PLU 4046 đã bán
* 4225 - tổng số bơ có mã PLU 4225 đã bán
* 4770 - tổng số bơ có mã PLU 4770 đã bán
* Total Bags – tổng số túi đã bán
* Small/Large/XLarge Bags – tổng số túi đã bán theo size""")
    st.write("""Có hai loại bơ trong tập dữ liệu và một số vùng khác nhau. Điều
này cho phép chúng ta thực hiện tất cả các loại phân tích cho
các vùng khác nhau, hoặc phân tích toàn bộ nước Mỹ theo một
trong hai loại bơ.""")

#=====================================================================================================
elif choice == "1.Average Price Prediction":
    st.subheader("1. ExtraTreesRegressor Models")
    # Upload file
    upload_file =st.file_uploader("Choose a file", type=['csv'])
    if upload_file is not None:
        data=pd.read_csv(upload_file)
        data.to_csv("avocado_new.csv", index= False)
        st.write("New data")
        st.dataframe(data.head(3))
    else:
        st.write("Original data")
        st.dataframe(data.head(3))
    # # xoá cột data thừa "unnamed"
    data =data.loc[:,~data.columns.str.contains('^Unnamed')]
    df=data.copy(deep= True)

    #hàm chuyển tháng qua mùa
    def convert_month(month):
        if month==3 or month==4 or month==5:
            return 0
        elif month==6 or month==7 or month==8:
            return 1
        elif month==9 or month==10 or month==11:
            return 2
        elif month==12 or month==1 or month==2:
            return 3
    #Chuyển dữ liệu date sang kiểu định dạng ngày
    df['Date'] = pd.to_datetime(df['Date'])
    #Thêm trường dữ liệu tháng
    df['Month'] = pd.DatetimeIndex(df['Date']).month
    #thêm trường dữ liệu Season dựa trên tháng
    df['Season'] = df['Month'].apply (lambda x: convert_month(x))
    
        # Label Encoder cho Type, year,month
    le= LabelEncoder()
    df['type_new']=le.fit_transform(df['type'])
    df['year_new']=le.fit_transform(df['year'])
    df['month_new']=le.fit_transform(df['Month'])
    #Onehot Encoder cho region
    df_ohe= pd.get_dummies(data=df,columns=['region'])
    X=df_ohe.drop(['Date','AveragePrice','4046','4225','4770','Small Bags','Large Bags','XLarge Bags','type','year','Month'],axis=1)
    y=df['AveragePrice']
    
    #-------------------------
    X_train,X_test,y_train,y_test =train_test_split(X,y,test_size= 0.25,random_state=0 )
    # vec = open("ExtraTreesRegressor.pkl", 'rb')
    # loaded_model = pickle.load(vec)

    pipe_line = Pipeline([('scaler',RobustScaler()),('a',ExtraTreesRegressor())])
    pipe_line.fit(X_train,y_train)
    y_pred= pipe_line.predict(X_test)
    r2score =r2_score(y_test,y_pred)
    maescore=mean_absolute_error(y_test,y_pred)
    #-------------------------

    # st.write("result of model after train & test above data:  r2 - score")
    # st.code(r2score)
    # st.write("result of model after train & test above data: mae - score")
    # st.code(maescore)

    
    #---------------- Prediction
    volume =st.slider("Total Volume",min_value=1,max_value= 70000000)
    st.write("Volume choose:", volume)
    bags =st.slider("Total Bags ",min_value=1,max_value= 20000000)
    st.write("No.bags choose:", bags)
    region_list= df['region'].unique()
    region = st.selectbox(
     'Where would you like to be predict?',(region_list)
     )
    st.write('Region selected:', region)
    type_list= df['type'].unique()
    type = st.selectbox(
     'Which avocado type would you like to be predict?',(type_list)
     )
    st.write('Avocado type selected:', type)
    Date=st.date_input('When would you like to be predict?')
    st.write('Date selected:', Date)
    #tạo dataframe mới để predict
    df = pd.DataFrame(data = [(volume,bags,Date,type,region)], columns = ['Total Volume', 'Total Bags', 'Date','type','region'])
    st.write("Data to predict:")
    st.dataframe(df.head(3))

    #Thêm trường dữ liệu tháng
    df['Month'] = pd.DatetimeIndex(df['Date']).month
    #Them trường dữ liệu year
    df['year'] = pd.DatetimeIndex(df['Date']).year

    #thêm trường dữ liệu Season dựa trên tháng
    df['Season'] = df['Month'].apply (lambda x: convert_month(x))
    
        # Label Encoder cho Type, year,month
    le= LabelEncoder()
    df['type_new']=le.fit_transform(df['type'])
    df['year_new']=le.fit_transform(df['year'])
    df['month_new']=le.fit_transform(df['Month'])
    #Onehot Encoder cho region
    df_ohe= pd.get_dummies(data=df,columns=['region'])
    X=df_ohe.drop(['Date','type','year','Month'],axis=1)

    remain_region=[]
    value=[]
    for i in region_list:
        if i !=region:
            a="region_"+i
            remain_region.append(a)
            value.append(0)
    remain=pd.DataFrame(value)
    remain=remain.T
    remain.columns =(remain_region)

    X = pd.merge(X, remain, left_index=True, right_index=True, how='left')


    #-------------------------
    # st.dataframe(X.head(3))
    y_pred= pipe_line.predict(X)
    
    st.write("Average Price Predict:")
    st.code(y_pred)
#=====================================================================================
elif choice == "2.Avocado Price by region - next years prediction":
    st.subheader("Avocado Price detail by region next years prediction ")
    upload_file =st.file_uploader("Choose a file", type=['csv'])

    if upload_file is not None:
        data=pd.read_csv(upload_file)
        data.to_csv("avocado_new.csv", index= False)
        st.write("New data")
        st.dataframe(data.head(3))
    else:
        st.write("Original data")
        st.dataframe(data.head(3))
    # # xoá cột data thừa "unnamed"
    df=data.copy(deep= True)
    region_list= df['region'].unique()
    region = st.selectbox(
     'Where would you like to be predict?',(region_list)
     )
    st.write('Region selected:', region)

    type_list= df['type'].unique()
    type = st.selectbox(
     'Which avocado type would you like to be predict?',(type_list)
     )
    st.write('Avocado type selected:', type)

    year_list= [1,2,3,4,5]
    year = st.selectbox(
     'How many year would you like to be predict?',(year_list)
     )
    st.write('Number of predict year selected:', year)

    # df_new=df[(df.type ==type) & (df.region ==region)][['Date','AveragePrice']]
    # df_new= df_new.groupby('Date').mean('AveragePrice')

    
    # f , ax =    plt.plot(df_new)
    # plt.title('Average Price of Avocado' )
    # st.pyplot(f)






    
    


    


    
    
    
    
    
    

# elif choice == "2.Oganic Avocado Price in Califonia prediction":

    











# # Filter Avocado - California
# # Make new dataframe from original dataframe: data
# df_ca = data[data['region'] == 'California']
# df_ca['Date'] = df_ca['Date'].str[:-3] 
# df_ca = df_ca[df_ca['type'] == 'organic']

# agg = {'AveragePrice': 'mean'}
# df_ca_gr = df_ca.groupby(df_ca['Date']).aggregate(agg).reset_index()
# df_ca_gr.head()

# df_ts = pd.DataFrame() 
# df_ts['ds'] = pd.to_datetime(df_ca_gr['Date']) 
# df_ts['y'] = df_ca_gr['AveragePrice'] 
# # head_=df_ts.head(3)
# # tail_=df_ts.tail(3)

# # Train/Test Prophet
# # create test dataset, remove last 10 months
# train = df_ts.drop(df_ts.index[-10:])
# test = df_ts.drop(df_ts.index[0:-10])

# # Build model
# model = Prophet(yearly_seasonality=True, \
#             daily_seasonality=False, weekly_seasonality=False) 
# model.fit(train)

# # 10 month in test and 12 month to predict new values
# months = pd.date_range('2017-06-01','2019-03-01', 
#               freq='MS').strftime("%Y-%m-%d").tolist()    
# future = pd.DataFrame(months)
# future.columns = ['ds']
# future['ds'] = pd.to_datetime(future['ds'])

# # Use the model to make a forecast
# forecast = model.predict(future)

# # calculate MAE/RMSE between expected and predicted values
# y_test = test['y'].values
# y_pred = forecast['yhat'].values[:10]
# mae_p = mean_absolute_error(y_test, y_pred)
# # print('MAE: %.3f' % mae_p)
# rmse_p = sqrt(mean_squared_error(y_test, y_pred))

# # visualization
# y_test_value = pd.DataFrame(y_test, index = pd.to_datetime(test['ds']),columns=['Actual'])
# y_pred_value = pd.DataFrame(y_pred, index = pd.to_datetime(test['ds']),columns=['Prediction'])

# # Long-term prediction for the next 5 years => Consider whether to expand cultivation/production, and trading
# m = Prophet(yearly_seasonality=True, \
#             daily_seasonality=False, weekly_seasonality=False) 
# m.fit(df_ts)
# future_new = m.make_future_dataframe(periods=12*5, freq='M') # next 5 years
# forecast_new = m.predict(future_new)





#     # st.text("Mean of Organic Avocado Average Price in California: " + str(round(df_ts['y'].mean(),2)) + "USD")
#     # st.write("""
#     # #### Build Model...""")
#     # st.write("""
#     # #### Caculate MAE/RMSE...""")
#     # st.code("MAE": str(round(mae_p,2)))
#     # st.code("RMSE": str(round(rmse_p,2)))
#     # st.write("""This result shows that Prophet's RMSE and MAE are good enough to predict the organic avocado AveragePrice in California, MAE = 0.16 (about 10% of the AveragePrice), compared to the AveragePrice ~ 1.68.""")
