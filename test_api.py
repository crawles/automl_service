import os

import numpy as np
import pandas as pd
import pytest
import requests

from sklearn.metrics import roc_auc_score

if os.environ.get('VCAP_SERVICES') is None: # running locally
    base_url = "http://0.0.0.0:8080"
else:                                       # running on CF
    base_url = "TBD"


def test_train_model():
    train_url = os.path.join(base_url, 'train_model')
    train_files = {'raw_data': open('data/data_train.json', 'rb'),
                   'labels' : open('data/label_train.json', 'rb'),
                   'params' : open('pipeline_parameters.yml', 'rb')}
    r = requests.post(train_url, files=train_files)
    assert r.status_code == 200

def test_serve_model():
    serve_url = os.path.join(base_url, 'serve_pred')
    test_files = {'raw_data': open('data/data_test.json', 'rb'),
                  'params' : open('pipeline_parameters.yml', 'rb')}
    r  = requests.post(serve_url, files=test_files)
    # parse result
    result = pd.DataFrame(r.json())
    result.index = result.index.astype(np.int)
    label_test = pd.read_json('data/label_test.json')
    result = result.loc[label_test.example_id]
    auc = roc_auc_score(label_test.label, result.values) 
    print "Test AUC: {}".format(auc)
    assert (auc > 0.9)
    
