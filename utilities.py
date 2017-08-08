# -*- coding: utf-8 -*-
"""
utilities.py
~~~~~~~~~~~~

Utility functions for automl service.
"""

import json

import pandas as pd
import yaml
import tsfresh

from tpot import TPOTClassifier
from tsfresh import extract_features, extract_relevant_features,\
                    select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters,\
                                       MinimalFCParameters


def read_file(request, fkey):
    """Given an input file, we want to read, parse, and load it 
    into a Pandas DataFrame"""
    d = request.files[fkey].read()
    j = json.loads(d)
    df = pd.DataFrame(j)
    return df

def read_params(request, fkey):
    """Parse parameter file"""
    d = request.files[fkey].read()
    return yaml.load(d)

def build_features(data, params):
    """Automated feature engineering function"""
    kwargs = params['extract_features']
    python_objects = ['impute_function', 'default_fc_parameters']
    for k in python_objects:
        kwargs[k] = eval(kwargs[k])
    return extract_features(data, **kwargs)

def train_model(X, y, params):
    """Train a sklearn-learn compatible classifier using AutoML via TPOT."""
    kwargs = params['tpot_classifier']
    tpot = TPOTClassifier(**kwargs)
    # remove this for testing, doesn't predict_proba
    if 'sklearn.svm.LinearSVC' in tpot.config_dict:
        del tpot.config_dict['sklearn.svm.LinearSVC']
    tpot.fit(X, y)
    return tpot
    

class Classifier(object):

    def __init__(self):
        self.cl = None

def load_module_from_string(module_name):
    __import__(module_name)
