import os

import numpy as np
import pandas as pd
import pytest
import requests

from sklearn.metrics import roc_auc_score

def test_train_model_1(host):
    '''Train sklearn model'''
    url = os.path.join(host, 'train_pipeline')
    train_files = {'raw_data': open('data/data_train.json', 'rb'),
                   'labels' : open('data/label_train.json', 'rb'),
                   'params' : open('parameters/train_parameters.yml', 'rb')}
    r  = requests.post(url,
            files=train_files)

def test_train_model_2(host):
    '''Train TPOT model'''
    url = os.path.join(host, 'train_pipeline')
    train_files = {'raw_data': open('data/data_train.json', 'rb'),
                   'labels' : open('data/label_train.json', 'rb'),
                   'params' : open('parameters/train_parameters_model2.yml', 'rb')}
    r  = requests.post(url,
            files=train_files)


def test_serve_model(host):
    serve_url = os.path.join(host, 'serve_prediction')
    test_files = {'raw_data': open('data/data_test.json', 'rb'),
                  'params' : open('parameters/test_parameters.yml', 'rb')}
    r  = requests.post(serve_url, files=test_files)

    # parse result
    result = pd.read_json(r.json()).set_index('id')
    result.index = result.index.astype(np.int)
    label_test = pd.read_json('data/label_test.json')
    result = result.loc[label_test.example_id]
    auc = roc_auc_score(label_test.label, result.values) 
    print "Test AUC: {}".format(auc)
    assert (auc > 0.9)
    
def test_serve_model_2(host):
    serve_url = os.path.join(host, 'serve_prediction')
    test_files = {'raw_data': open('data/data_test.json', 'rb'),
                  'params' : open('parameters/test_parameters_model2.yml', 'rb')}
    r  = requests.post(serve_url, files=test_files)

    # parse result
    result = pd.read_json(r.json()).set_index('id')
    result.index = result.index.astype(np.int)
    label_test = pd.read_json('data/label_test.json')
    result = result.loc[label_test.example_id]
    auc = roc_auc_score(label_test.label, result.values) 
    print "Test AUC: {}".format(auc)
    assert (auc > 0.9)

def test_get_models(host):
    '''Show all available models'''
    url = os.path.join(host, 'models')
    r  = requests.get(url)
    assert r.status_code == 200

