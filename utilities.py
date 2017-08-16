# -*- coding: utf-8 -*-
"""
utilities.py
~~~~~~~~~~~~

Utility functions for automl service.
"""
import copy
import json

import pandas as pd
import yaml
import sklearn
import tpot
import tsfresh

from sklearn.ensemble import RandomForestClassifier
from tsfresh import extract_features, extract_relevant_features,\
                    select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters,\
                                       MinimalFCParameters


#TODO: change name to load json
def read_file(d):
    """Given an input file, we want to read, parse, and load it 
    into a Pandas DataFrame"""
    j = json.loads(d)
    df = pd.DataFrame(j)
    return df

def read_params(d):
    """Parse parameter file"""
    return yaml.load(d)

def build_features(data, params):
    """Automated feature engineering function"""
    kwargs = params['extract_features']
    python_objects = ['impute_function', 'default_fc_parameters']
    for k in python_objects:
        kwargs[k] = eval(kwargs[k])
    return extract_features(data, **kwargs)

def train_model(X, y, params):
    """Train a sklearn-learn compatible classifier ."""
    model_params = params['model_training']
    python_objects = ['model']
    for k in python_objects:
        model_params[k] = eval(model_params[k])
    model = model_params['model']
    kwargs = model_params['model_args']
    if kwargs:
        cl = model(**kwargs)
    else:
        cl = model()
    cl.fit(X, y)
    return cl
    
def cross_validate(cl, X_train, y_train):
    scoring = ['roc_auc', 'accuracy']
    cv = sklearn.model_selection.cross_validate(cl, X_train, y_train,
                                                cv=5, scoring=scoring)
    mean_accuracy = cv['test_accuracy'].mean()
    mean_roc_auc = cv['test_roc_auc'].mean()
    return (mean_accuracy, mean_roc_auc)

class ModelFactory(object):

    def __init__(self):
        self.pipelines = dict()
        self.cl = None

    def __getitem__(self, item):
        return self.pipelines[item]

    def add_pipeline(self, cl, params):
        pipeline_id = params['pipeline_id']
        self.pipelines[pipeline_id] = dict()
        pipeline = self.pipelines[pipeline_id]
        pipeline['extract_features'] = params['extract_features']
        pipeline['model'] = cl

    def use_pipeline(self, df, pipeline_id): 
        params = self.pipelines[pipeline_id]
        X = extract_features(df, **params['extract_features'])
        cl = params['model']
        scores = cl.predict_proba(X)[:,1]
        result = pd.DataFrame(scores,
                              columns=['score'],
                              index=X.index)
        return result
        

def load_module_from_string(module_name):
    __import__(module_name)
