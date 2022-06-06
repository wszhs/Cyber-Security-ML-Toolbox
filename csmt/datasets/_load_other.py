import csmt.Interpretability.sage as sage
from csmt.datasets._base import get_mask,get_true_mask
import pandas as pd
import re
import gender_guesser.detector as detector

def load_bike():
    df = sage.datasets.bike()
    feature_names = df.columns.tolist()[:-3]
    return df,feature_names

def load_bank():
    df = sage.datasets.bank()
    feature_names = df.columns.tolist()[:-1]
    return df,feature_names

def load_airbnb():
    df = sage.datasets.airbnb()
    # Categorical features
    categorical_columns = ['neighbourhood_group', 'neighbourhood', 'room_type']
    for column in categorical_columns:
        df[column] = pd.Categorical(df[column]).codes
    df = df[df['price'] < df['price'].quantile(0.995)]
    # Features derived from name
    df['name_length'] = df['name'].apply(lambda x: len(x))
    df['name_isupper'] = df['name'].apply(lambda x: int(x.isupper()))
    df['name_words'] = df['name'].apply(lambda x: len(re.findall(r'\w+', x)))
    # Host gender guess
    guesser = detector.Detector()
    df['host_gender'] = df['host_name'].apply(lambda x: guesser.get_gender(x.split(' ')[0]))
    df['host_gender'] = pd.Categorical(df['host_gender']).codes
    # Number of days since last review
    most_recent = df['last_review'].max()
    df['last_review'] = (most_recent - df['last_review']).dt.days
    df['last_review'] = (df['last_review'] - df['last_review'].mean()) / df['last_review'].std()
    df['last_review'] = df['last_review'].fillna(-5)
    # Missing values
    df['reviews_per_month'] = df['reviews_per_month'].fillna(0)
    # Normalize other numerical features
    df['number_of_reviews'] = (df['number_of_reviews'] - df['number_of_reviews'].mean()) / df['number_of_reviews'].std()
    df['availability_365'] = (df['availability_365'] - df['availability_365'].mean()) / df['availability_365'].std()
    df['name_length'] = (df['name_length'] - df['name_length'].mean()) / df['name_length'].std()

    # Normalize latitude and longitude
    df['latitude'] = (df['latitude'] - df['latitude'].mean()) / df['latitude'].std()
    df['longitude'] = (df['longitude'] - df['longitude'].mean()) / df['longitude'].std()
    # Drop columns
    df = df.drop(['id', 'host_id', 'host_name', 'name'], axis=1)

    target_col = 'price'
    cols = df.columns.tolist()
    del cols[cols.index(target_col)]
    cols.append(target_col)
    feature_names = cols[:-1]
    return df,feature_names
