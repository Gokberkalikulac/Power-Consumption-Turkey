import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
import itertools
import warnings
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
color = sns.color_palette()
import matplotlib as mpl
mpl.use('macosx')
sns.set_style('darkgrid')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
import mylibrary
warnings.filterwarnings('ignore')
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

################################################################
#Forecasting power demand accurately can yield significant benefits for various stakeholders, whether at a national, municipal, or household level.
# Anticipating power requirements with precision empowers stakeholders to optimize production strategies, thereby reducing costs.
# Additionally, they can make informed decisions regarding purchasing energy from external sources to meet demand efficiently.
# Notably, in contexts like daily energy exchange tendering processes, stakeholders stand to enhance profitability through precise forecasting.
# Within this notebook, I will elucidate fundamental principles for training a Machine Learning model to predict Turkey's power consumption.
# I use different approaches for predicting the future electrical consumption of Turkey.
################################################################
#Links to Datasets
# https://www.kaggle.com/datasets/dharanikra/electrical-power-demand-in-turkey/data
# https://www.kaggle.com/datasets/hgultekin/hourly-power-consumption-of-turkey-20162020


################################################################
#################### Importing Datasets ########################
################################################################
df1 = mylibrary.load("Consumption.csv")
df2 = mylibrary.load("GenerationandConsumption.csv")

################################################################
################ Processing The Datasets #######################
################################################################
df1 = mylibrary.processing_dataframe1(df1)
df2 = mylibrary.processing_dataframe2(df2)

df1 = df1.loc[df1["Date"] <= "2019-12-31"]

mylibrary.check_df(df1)
mylibrary.check_df(df2)

################################################################
################ Find Zeros in DataFrame and solve #############
################################################################
mylibrary.solution_for_zero_indeces(df1,df2)
################################################################

################################################################
########## Resammpling and Cancating The Data Frames ###########
################################################################
df1 = df1.set_index("Date").resample("H").mean()
df2 = df2.set_index("Date").resample("H").mean()

df = pd.concat([df1, df2])

################################################################
######################## Visualization #########################
################################################################
df.reset_index(inplace=True)

# Hourly Consumption
fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(x=df["Date"].dt.hour, y="Consumption (MWh)", data=df)
ax.set_title('Average Hourly Consumption', fontsize=11)
ax.set_xlabel('Hour of the Day')
ax.set_ylabel('Consumption (MWh)')
plt.show()

# Daily Consumption
fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(x=df["Date"].dt.day, y="Consumption (MWh)", data=df)
ax.set_title('Average Daily Consumption', fontsize=11)
ax.set_xlabel('Day of the Month')
ax.set_ylabel('Consumption (MWh)')
plt.show()

# Weekly Consumption
fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(x=df['Date'].dt.isocalendar().week, y="Consumption (MWh)", data=df)
ax.set_title('Average Weekly Consumption', fontsize=11)
ax.set_xlabel('Week of the Year')
ax.set_ylabel('Consumption (MWh)')
plt.show()

# Monthly Consumption
fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(x=df["Date"].dt.month, y="Consumption (MWh)", data=df)
ax.set_title('Average Monthly Consumption', fontsize=11)
ax.set_xlabel('Month of the Year')
ax.set_ylabel('Consumption (MWh)')
plt.show()


################################################################
###############  EDA(Exploratory Data Analysis) ################
################################################################
df.describe().T
mylibrary.check_outlier(df,"Consumption (MWh)")
mylibrary.check_df(df)
mylibrary.replace_with_thresholds(df,"Consumption (MWh)")
low_limit, up_limit = mylibrary.outlier_thresholds(df, "Consumption (MWh)")
df.describe().T
# There is no Duplicate Date
df = df.set_index("Date")
df.index.duplicated().sum()
dff = df.copy()

#Resampling the dataframe
df_month = df.resample("M").mean()
df_week = df.resample("W").mean()
df_day = df.resample("D").mean()
df_year = df.resample("Y").mean()
mylibrary.replace_with_thresholds(df_week,"Consumption (MWh)")
mylibrary.replace_with_thresholds(df_month,"Consumption (MWh)")
mylibrary.replace_with_thresholds(df_day,"Consumption (MWh)")
mylibrary.replace_with_thresholds(df_year,"Consumption (MWh)")
mylibrary.check_outlier(df_month,"Consumption (MWh)")
mylibrary.check_outlier(df_week,"Consumption (MWh)")
mylibrary.check_outlier(df_day,"Consumption (MWh)")
mylibrary.check_outlier(df_year,"Consumption (MWh)")

################################################################
# Until this step prosses data, analyse data and check if any outliers or null values.
################################################################

################################################################
########### Time Series Components and Stationary Test #########
#################### Stationary Test ###########################
################################################################
# With this below code we analyse the behavior of the stationary according to resampled dataframes
def is_stationary(y):
    # "HO: Non-stationary"
    # "H1: Stationary"
    p_value = sm.tsa.stattools.adfuller(y)[1]
    if p_value < 0.05:
        print(F"Result: Stationary (H0: non-stationary, p-value: {round(p_value, 3)})")
    else:
        print(F"Result: Non-Stationary (H0: non-stationary, p-value: {round(p_value, 3)})")
is_stationary(df)
is_stationary(df_day)
is_stationary(df_week)
is_stationary(df_month)
is_stationary(df_year)


# Ploting the resampled dataframe in order see stationary or non-stationary better
df.plot(figsize = (16, 6), color=color[1], title="Energy Consumption in Turkey Hourly")
df_day.plot(figsize = (16, 6), color=color[1], title="Energy Consumption in Daily")
df_week.plot(figsize = (16, 6), color=color[1], title="Energy Consumption in Weekly")
df_month.plot(figsize = (16, 6), color=color[1], title="Energy Consumption in Monthly")
df_year.plot(figsize = (16, 6), color=color[1], title="Energy Consumption in Yearly")


################################################################
################## Time Series Visualization ###################
################################################################
fig, axes = plt.subplots(2, 2, figsize=(16, 8))

# Energy Consumption in Turkey Hourly
df.plot(ax=axes[0, 0], color='blue')
axes[0, 0].set_title("Energy Consumption in Turkey Hourly")
axes[0, 0].set_ylabel("Consumption (MWh)")

# Energy Consumption Daily
df_day.plot(ax=axes[0, 1], color='red')
axes[0, 1].set_title("Energy Consumption Daily")
axes[0, 1].set_ylabel("Consumption (MWh)")

# Energy Consumption Weekly
df_week.plot(ax=axes[1, 0], color='green')
axes[1, 0].set_title("Energy Consumption Weekly")
axes[1, 0].set_ylabel("Consumption (MWh)")

# Energy Consumption Monthly
df_month.plot(ax=axes[1, 1], color='orange')
axes[1, 1].set_title("Energy Consumption Monthly")
axes[1, 1].set_ylabel("Consumption (MWh)")
plt.tight_layout()
plt.show()

################################################################
################ Time Series Decompose #########################
################################################################
result = seasonal_decompose(df_month["Consumption (MWh)"], model='add', period=12)
result.plot()
plt.show()

################################################################
################### Train-Test Split ###########################
################################################################
# First I divide data set train and test set partially(app % 77)
train = df.loc[df.index < "2021-06-01"]["Consumption (MWh)"]
test = df.loc[df.index > "2021-06-01"]["Consumption (MWh)"]

fig, ax = plt.subplots(figsize=(12,6))
train.plot(ax=ax,label="Training Set",legend=True)
test.plot(ax=ax, label="Test Set",legend=True)
ax.axvline("2021-06-01", color='black', linestyle='--')
ax.set_xlabel("Date")
ax.set_ylabel("Power Consumption (MWh)")
ax.set_title("Monthly Consumption by Hours")
plt.show()

# As we can see from hourly train and test split we should resample them I choose monthly and visialize them.
train_month = train.resample("M").mean()
test_month = test.resample("M").mean()

train_week = train.resample("W").mean()
test_week = test.resample("W").mean()

fig, ax = plt.subplots(figsize=(12,6))
train_month.plot(ax=ax,label="Training Set",legend=True)
test_month.plot(ax=ax, label="Test Set",legend=True)
ax.axvline("2021-06-01", color='black', linestyle='--')
ax.set_xlabel("Date")
ax.set_ylabel("Power Consumption (MWh)")
ax.set_title("Monthly Consumption by Years")
plt.show()


################################################################
################### Smoothing Methods ##########################
################################################################
# At this step I use Expotential Smooting Method. Because dataset have attributes of
# Non-stationarty, seasonability, and trend as we checked before. Also used optimizer
# for finding best a,b,g values.
################################################################

alphas = betas = gammas = np.arange(0.10, 1, 0.05)
abg = list(itertools.product(alphas, betas, gammas))

def tes_optimizer(train, abg, step):
    best_alpha, best_beta, best_gamma, best_mae = None, None, None, float("inf")
    for comb in abg:
        tes_model = ExponentialSmoothing(train, seasonal="add", seasonal_periods=12).\
            fit(smoothing_level=comb[0], smoothing_slope=comb[1], smoothing_seasonal=comb[2])
        y_pred = tes_model.forecast(step)
        if np.isnan(y_pred).any():  # Eğer y_pred içerisinde NaN değerler varsa
            continue  # Devam et, bir sonraki iterasyona geç
        mae = mean_absolute_error(test_month, y_pred)
        if mae < best_mae:
            best_alpha, best_beta, best_gamma, best_mae = comb[0], comb[1], comb[2], mae
        print([round(comb[0], 2), round(comb[1], 2), round(comb[2], 2), round(mae, 2)])

    print("best_alpha:", round(best_alpha, 2), "best_beta:", round(best_beta, 2), "best_gamma:", round(best_gamma, 2),
          "best_mae:", round(best_mae, 4))

    return best_alpha, best_beta, best_gamma, best_mae

best_alpha, best_beta, best_gamma, best_mae = tes_optimizer(train_month, abg, step=len(test_month))

################################################################
# best_alpha: 0.3 best_beta: 0.1 best_gamma: 0.45 best_mae: 994.1681
################################################################

tes_model = ExponentialSmoothing(train_month, seasonal="add", seasonal_periods=12).\
            fit(smoothing_level=best_alpha, smoothing_slope=best_beta, smoothing_seasonal=best_gamma)

y_pred = tes_model.forecast(len(test_month))
################################################################
# y_pred:
#2021-06-30    36966.200300
#2021-07-31    40524.491066
#2021-08-31    39557.265249
#2021-09-30    38022.304712
#2021-10-31    34888.670509
#2021-11-30    36539.950131
#2021-12-31    37591.025286
################################################################
def plot_prediction(y_pred, label):
    train_month.plot(legend=True, label="TRAIN")
    test_month.plot(legend=True, label="TEST")
    y_pred.plot(legend=True, label="PREDICTION")
    plt.ylabel("Monthly Consumption")
    plt.title("Train, Test and Predicted Test Using "+label)
    plt.show()

plot_prediction(y_pred, "Triple Exponential Smoothing")

################################################################
################Future Forecasting Wtih Final Model TES ############
################################################################

tes_model_final = ExponentialSmoothing(df_month, seasonal="add", seasonal_periods=12).\
            fit(smoothing_level=best_alpha, smoothing_slope=best_beta, smoothing_seasonal=best_gamma)

future = tes_model_final.forecast(24)

def plot_future(future, label):
    df_month.plot(legend=True, label="Dataset")
    future.plot(legend=True, label="Future")
    plt.title("Predicted Future Using "+label)
    plt.ylabel("Monthly Consumption")
    plt.show()

plot_future(future,"Triple Exponential Smoothing")

################################################################
#################### Statistical Models ########################
################################################################

################################################################
########################## SARIMA ##############################
################################################################
p = d = q = range(0, 3)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

def sarima_optimizer_mae(train_day, pdq, seasonal_pdq):
    best_mae, best_order, best_seasonal_order = float("inf"), float("inf"), None

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                model = SARIMAX(train_day, order=param, seasonal_order=param_seasonal,trend=None)
                sarima_model = model.fit(disp=0)
                y_pred_test = sarima_model.get_forecast(steps=len(test_month))
                y_pred = y_pred_test.predicted_mean
                mae = mean_absolute_error(test_month, y_pred)
                if mae < best_mae:
                    best_mae, best_order, best_seasonal_order = mae, param, param_seasonal
                print('SARIMA{}x{}12 - MAE:{}'.format(param, param_seasonal, mae))
            except:
                continue
    print('SARIMA{}x{}12 - MAE:{}'.format(best_order, best_seasonal_order, best_mae))
    return best_order, best_seasonal_order

best_order, best_seasonal_order = sarima_optimizer_mae(train_month, pdq, seasonal_pdq)

################################################################
# Best Parameters : ((1, 2, 1), (1, 1, 1, 12))
################################################################

model = SARIMAX(train_month, order=best_order, seasonal_order=best_seasonal_order,trend=None)
sarima_final_model = model.fit(disp=0)
y_pred_test = sarima_final_model.get_forecast(steps=len(test_month))
y_pred = y_pred_test.predicted_mean
mean_absolute_error(test_month, y_pred)

################################################################
# Mae : 921.8699392068631
################################################################
def plot_prediction(y_pred, label):
    train_month.plot(legend=True, label="TRAIN")
    test_month.plot(legend=True, label="TEST")
    y_pred.plot(legend=True, label="PREDICTION")
    plt.ylabel("Monthly Consumption")
    plt.title("Train, Test and Predicted Test Using "+label)
    plt.show()

plot_prediction(pd.Series(y_pred, index=test_month.index), "SARIMA")



################################################################
############ Future Forecasting Wtih Final Model SARIMA ########
################################################################

final_model = SARIMAX(df_month, order=(1,0,1), seasonal_order=(1, 0, 1, 12),trend=None)
sarima_final_model = final_model.fit(disp=0)
y_pred_test = sarima_final_model.get_forecast(steps=24)
future = y_pred_test.predicted_mean

def plot_future(future, label):
    df_month.plot(legend=True, label="Dataset")
    future.plot(legend=True, label="Future")
    plt.ylabel("Monthly Consumption")
    plt.title("Predicted Future Using "+label)
    plt.show()

plot_future(future,"Sarima")



################################################################
##Normal Approach (Feature Engineering and Machine Learning)####
################################################################

################################################################
##################### Featute Engineering ######################
################################################################
df = dff.copy()
df.reset_index(inplace=True)

def create_date_features(df):
    df['month'] = df.Date.dt.month
    df['day_of_month'] = df.Date.dt.day
    df['hour_of_day'] = df.Date.dt.hour
    df['quarter'] = df.Date.dt.quarter
    df['day_of_year'] = df.Date.dt.dayofyear
    df['week_of_year'] = df.Date.dt.weekofyear
    df['day_of_week'] = df.Date.dt.dayofweek
    df['year'] = df.Date.dt.year
    df["is_wknd"] = df.Date.dt.weekday // 4
    df['is_month_start'] = df.Date.dt.is_month_start.astype(int)
    df['is_month_end'] = df.Date.dt.is_month_end.astype(int)
    return df

df = create_date_features(df)
df.head()


################################################################
##################### Adding Random Noise ######################
################################################################
# In order to overfitting I have added random noise

def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=(len(dataframe),))


################################################################
################## Lag/Shifted Features ########################
################################################################

def lag_features(dataframe, lags):
    for lag in lags:
        dataframe['consumption_lag_' + str(lag)] = dataframe["Consumption (MWh)"].transform(
            lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe

df = lag_features(df, [31, 99, 124, 150, 185, 218, 243, 300, 546, 728])
mylibrary.check_df(df)



########################
# Rolling Mean Features
########################
def roll_mean_features(dataframe, windows):
    for window in windows:
        dataframe['consumption_roll_mean_' + str(window)] = dataframe["Consumption (MWh)"].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=10, win_type="triang").mean()) + random_noise(
            dataframe)
    return dataframe


df = roll_mean_features(df, [365, 546])


########################
# Exponentially Weighted Mean Features
########################

def ewm_features(dataframe, alphas, lags):
    for alpha in alphas:
        for lag in lags:
            dataframe['consumption_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe["Consumption (MWh)"].transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe

alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
lags = [31, 99, 124, 150, 185, 218, 243, 300, 546, 728]

df = ewm_features(df, alphas, lags)


########################
# One-Hot Encoding
########################

df = pd.get_dummies(df,columns=['day_of_week', 'month'])


########################
# Converting sales to log(1+sales)
########################

df['Consumption (MWh)'] = np.log1p(df["Consumption (MWh)"].values)

########################
# Time-Based Validation Sets
########################
train = df.loc[df["Date"] < "2021-06-01"]
test = df.loc[df["Date"] >= "2021-06-01"]

cols = [col for col in train.columns if col not in ["Consumption (MWh)","Date"]]

y_train = train['Consumption (MWh)']
X_train = train[cols]

y_test = test['Consumption (MWh)']
X_test = test[cols]

y_train.shape, X_train.shape, X_test.shape, y_test.shape



from xgboost import XGBRegressor
xgb_reg = XGBRegressor(n_estimators=10000,early_stopping_rounds=50,learning_rate=0.01)
xgb_reg.fit(X_train,y_train,eval_set=[(X_train,np.expm1(y_train)), (X_test,np.expm1(y_test))],verbose=100)

def plot_xgb_importances(m, plot=False, num=10):
    gain = m.get_booster().get_score(importance_type='gain')
    feat_imp = pd.DataFrame({'feature': list(gain.keys()),
                             'split': m.get_booster().get_score(importance_type='weight').values(),
                             'gain': list(gain.values())}).sort_values('gain', ascending=False)
    if plot:
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="gain", y="feature", data=feat_imp.head(25))
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.show()
    else:
        print(feat_imp.head(num))
    return feat_imp


plot_xgb_importances(xgb_reg, num=20, plot=True)


###Forecasting ####

predict = np.expm1(xgb_reg.predict(X_test))

dff = pd.DataFrame({"Date": df["Date"],"Real": np.expm1(df["Consumption (MWh)"])})

dff_predict = pd.DataFrame({"Date":test["Date"],"Prediction":predict})

merged_df = pd.merge(dff, dff_predict, on='Date', how='left')


# Görseli oluştur

df.set_index('Date', inplace=True)

plt.figure(figsize=(16, 5))
plt.plot(merged_df["Date"],merged_df["Real"] , label='Real Data')
plt.plot(merged_df["Date"], merged_df["Prediction"], label='Predictions', color='orange')
plt.title("Real Data and Prediction")
plt.xlabel("Date")
plt.ylabel("Consumption (MWh)")
plt.legend()
plt.show()
df
score = np.sqrt(mean_squared_error(np.expm1(test["Consumption (MWh)"]),predict))
mae = mean_absolute_error(np.expm1(test["Consumption (MWh)"]),predict)

future_data_point = pd.to_datetime("2023-01-01 01:00:00")

# Bu veri noktasını Pandas DataFrame'e dönüştürme
future_data = pd.DataFrame({"Date": [future_data_point]})

future_predictions = xgb_reg.predict(future_data)
