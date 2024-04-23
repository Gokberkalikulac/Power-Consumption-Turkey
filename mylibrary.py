import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

#############Reading Dataframe################
def load(adress):
    data = pd.read_csv(adress)
    return data
##############################################

#############Understanding the Data###########
def check_df(dataframe, head=5, tail=5, quan=False):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(tail))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Counts #####################")
    print(dataframe.value_counts())

##############################################

#################Preprocessing for DataFrame1################
def processing_dataframe1(df):
    df['Date'] = pd.to_datetime(df['Date'] + " " + df["Hour"])
    df.drop('Hour', axis=1, inplace=True)
    df['Consumption (MWh)'] = df['Consumption (MWh)'].str.replace(',', '')
    df['Consumption (MWh)'] = pd.to_numeric(df['Consumption (MWh)'])
    df = df.sort_values('Date')
    df.reset_index(drop=True, inplace=True)
    return df
##############################################


#################Preprocessing for DataFrame2################
def processing_dataframe2(df):
    df = df[["Date_Time", "Consumption (MWh)"]]
    df.columns = ["Date", "Consumption (MWh)"]
    df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y %H:%M')
    df = df.sort_values('Date')
    df.reset_index(drop=True, inplace=True)
    return df
##############################################

###########Zero Indeces##########################

def solution_for_zero_indeces(df1,df2):
    zero_indices = df1[df1["Consumption (MWh)"] == 0].index

    for index in zero_indices:
        previous_value = df1.loc[index - 1, "Consumption (MWh)"]
        next_value = df1.loc[index + 1, "Consumption (MWh)"]
        mean_value = (previous_value + next_value) / 2
        df1.at[index, "Consumption (MWh)"] = mean_value

    zero_indices = df2[df2["Consumption (MWh)"] == 0].index



#############Aykırı Gözlem Analizi##########
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit