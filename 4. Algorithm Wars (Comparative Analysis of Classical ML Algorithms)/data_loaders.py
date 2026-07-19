import pandas as pd
from datasets import load_dataset
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import fetch_openml

def get_data1():
    data = load_breast_cancer(as_frame=True)

    X = data.data
    y = data.target

    numerical_columns = data['feature_names'].tolist()
    categorical_columns = []

    return X, y, numerical_columns, categorical_columns, 'tabular'

def get_data2():
    adult = fetch_openml('adult', version=2, as_frame=True)

    X = adult.data
    y = (adult.target == '>50K').astype(int)

    numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()

    return X, y, numerical_columns, categorical_columns, 'tabular'

def get_data3():
    ds = load_dataset("SetFit/bbc-news")
    full_df = pd.concat([ds['train'].to_pandas(), ds['test'].to_pandas()])

    X = full_df['text']    # select the text column, not the whole DataFrame
    y = full_df['label']
    
    return X, y, [], [], 'text'

