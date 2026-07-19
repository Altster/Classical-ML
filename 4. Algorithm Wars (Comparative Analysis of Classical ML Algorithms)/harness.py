import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer

from models import get_models

import random
import numpy as np

random.seed(37)
np.random.seed(37)

def compare_models(X, y, num_clmns, cat_clmns, preprocessor_type='tabular', cv_folds=5, scoring='accuracy'):
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=37)
    preprocessor = get_preprocessor(preprocessor_type, num_clmns, cat_clmns)

    models = get_models(preprocessor_type)

    rows = []
    for name, model in models.items():

        modelpipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        results = cross_validate(modelpipe, X, y, scoring=scoring, cv=cv, n_jobs=-1, return_train_score=True)

        rows.append({
            'model': name,
            'mean_train_score': results['train_score'].mean() * 100,
            'mean_test_score': results['test_score'].mean() * 100,
            'std_test_score': results['test_score'].std() * 100,
            'mean_fit_time': results['fit_time'].mean(),
            'mean_score_time': results['score_time'].mean(),
        })

    results_df = pd.DataFrame(rows).sort_values('mean_test_score', ascending=False).reset_index(drop=True)

    return results_df


def get_preprocessor(type, num_clmns, cat_clmns):
    if type == 'tabular':
        numerical_transformer = Pipeline(steps = [
            ('imputer', SimpleImputer(strategy = 'median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps = [
            ('imputer', SimpleImputer(strategy = 'most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer(transformers = [
            ('numerical', numerical_transformer, num_clmns),
            ('categorical', categorical_transformer, cat_clmns)
        ])

    elif type == 'text':
        preprocessor = TfidfVectorizer(
            max_features=20_000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            stop_words="english",
            sublinear_tf=True
        )

    return preprocessor