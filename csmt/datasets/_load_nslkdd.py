'''
Author: your name
Date: 2021-03-25 14:29:50
LastEditTime: 2021-04-01 16:10:41
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /NashHE/nashhe/datasets/_load_cicnslkdd.py
'''
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
import numpy as np
from os import path

def load_nslkdd_old():
    col_names = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment",
             "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted",
             "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
             "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
             "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
             "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
             "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
             "dst_host_srv_rerror_rate", "class"]

    train_dataset_path= 'csmt/datasets/data/NSL-KDD/KDDTrain+.csv'
    test_dataset_path = 'csmt/datasets/data/NSL-KDD/KDDTest+.csv'
    train_dataframe='csmt/datasets/pickles/nslkdd_dataframe_train.pkl'
    test_dataframe='csmt/datasets/pickles/nslkdd_dataframe_test.pkl'

    # if path.exists(train_dataframe) and path.exists(test_dataframe):

    #     df_train = pd.read_pickle(train_dataframe)
    #     df_test = pd.read_pickle(test_dataframe)
    #     X_train = df_train.drop(['class'], axis=1)
    #     y_train = df_train['class']
    #     X_test = df_test.drop(['class'], axis=1)
    #     y_test = df_test['class']
    #     return X_train,y_train,X_test,y_test
    
    df_train = pd.read_csv(train_dataset_path, header=0, names=col_names)
    df_test = pd.read_csv(test_dataset_path, header=0, names=col_names)
    
    df_train = df_train.dropna()
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.dropna()
    df_test = df_test.reset_index(drop=True)

    # Identify caategoricaal features
    for col_name in df_train.columns:
        if df_train[col_name].dtype == 'object':
            unique_category = len(df_train[col_name].unique())
            # print("Training Dataset: Feature '{col_name}' has {unique_cat} categories".format(col_name=col_name,
            #                                                                                   unique_cat=unique_category))
    for col_name in df_test.columns:
        if df_test[col_name].dtype == 'object':
            unique_category = len(df_test[col_name].unique())
            # print("Testing Dataset: Feature '{col_name}' has {unique_cat} categories".format(col_name=col_name,
            #                                                                                  unique_cat=unique_category))

    categorical_columns = ['protocol_type', 'service', 'flag']
    df_train_categorical_values = df_train[categorical_columns]
    df_test_categorical_values = df_test[categorical_columns]

    # one-hot-encoding, create dummy column names
    unique_protocal = sorted(df_train['protocol_type'].unique())
    crafted_protocal = ['Protocol_type_' + x for x in unique_protocal]

    unique_service = sorted(df_train['service'].unique())
    crafted_service = ['service_' + x for x in unique_service]

    unique_flag = sorted(df_train['flag'].unique())
    crafted_flag = ['flag_' + x for x in unique_flag]

    train_dummy_cols = crafted_protocal + crafted_service + crafted_flag

    unique_service_test = sorted(df_test['service'].unique())
    crafted_service_test = ['service_' + x for x in unique_service_test]
    test_dummy_cols = crafted_protocal + crafted_service_test + crafted_flag

    df_train_categorical_value_encode = df_train_categorical_values.apply(LabelEncoder().fit_transform)
    df_test_categorical_value_encode = df_test_categorical_values.apply(LabelEncoder().fit_transform)

    oneHotEncoder = OneHotEncoder()
    df_train_categorical_values_onehot = oneHotEncoder.fit_transform(df_train_categorical_value_encode)
    df_train_cat_data = pd.DataFrame(df_train_categorical_values_onehot.toarray(), columns=train_dummy_cols)

    # feature test in service miss 6 categories
    train_service = df_train['service'].tolist()
    test_service = df_test['service'].tolist()
    service_difference = list(set(train_service) - set(test_service))
    service_difference = ['service_' + x for x in service_difference]

    df_test_categorical_values_onehot = oneHotEncoder.fit_transform(df_test_categorical_value_encode)
    df_test_cat_data = pd.DataFrame(df_test_categorical_values_onehot.toarray(), columns=test_dummy_cols)

    for col in service_difference:
        df_test_cat_data[col] = 0

    # join and replace original dataset
    new_df_train = df_train.join(df_train_cat_data)
    new_df_train.drop('flag', axis=1, inplace=True)
    new_df_train.drop('protocol_type', axis=1, inplace=True)
    new_df_train.drop('service', axis=1, inplace=True)

    new_df_test = df_test.join(df_test_cat_data)
    new_df_test.drop('flag', axis=1, inplace=True)
    new_df_test.drop('protocol_type', axis=1, inplace=True)
    new_df_test.drop('service', axis=1, inplace=True)


    new_df_train['class'] = new_df_train['class'].map(lambda x: 0 if x == "normal" else 1)
    new_df_test['class'] = new_df_test['class'].map(lambda x: 0 if x == "normal" else 1)
    print(new_df_test['class'].value_counts())
    print(new_df_train['class'].value_counts())

    # new_df_train.to_pickle(train_dataframe)
    # new_df_test.to_pickle(test_dataframe)

    X_train = new_df_train.drop(['class'], axis=1)
    y_train = new_df_train['class']
    # X_test = new_df_test.drop(['class'], axis=1)
    # y_test = new_df_test['class']
    X=X_train
    y=y_train
    mask=get_true_mask([column for column in X])
    return X,y,mask

def load_nslkdd():
    from csmt.datasets._base import get_mask,get_true_mask,add_str,get_dict,train_val_test_split
    columns = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
        "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root", 
        "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
        "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
        "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", 
        "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
        "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"]
    df = pd.read_csv("csmt/datasets/data/NSL-KDD/kddcup.data.corrected", sep=",", names=columns, index_col=None)
    df = df[df["service"] == "http"]
    df = df.drop("service", axis=1)
    # print(df['label'].value_counts())
    # print(df[0:10])
    df['label'] = df['label'].map(lambda x: 1 if x != "normal." else x)
    df['label'] = df['label'].map(lambda x: 0 if x == "normal." else x)
    print(df['label'].value_counts())
    for col in df.columns:
        if df[col].dtype == "object":
            encoded = LabelEncoder()
            encoded.fit(df[col])
            df[col] = encoded.transform(df[col])
    X=df.iloc[:,0:-1]
    y=df.iloc[:,-1]
    mask=get_true_mask([column for column in X])
    return X,y,mask