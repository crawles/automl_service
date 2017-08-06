# -*- coding: utf-8 -*-
"""
automl_service.py
~~~~~~~~~~~~~~~~~

App implements an automl pipeline.

"""

import os
import json

from flask import Flask, request, jsonify
import pandas as pd
import requests
import sklearn

from utilities import build_features, read_file, read_params,\
        train_model, Classifier


app = Flask(__name__)

classifier = Classifier()

# curl -F "raw_data=@data/data_train.json" -F "labels=@data/label_train.json" -F "params=@pipeline_parameters.yml" -X POST http://0.0.0.0:8080/train_model
@app.route('/train_model', methods=['POST'])
def train_api():
    df = read_file(request, 'raw_data')
    y = read_file(request, 'labels')
    params = read_params(request, 'params')
    X = build_features(df, params)
    cl = train_model(X, y.label, params)
    classifier.cl = cl
    return str(cl.fitted_pipeline_)


@app.route('/serve_pred', methods=['POST'])
def serve_api():
    df = read_file(request, 'raw_data')
    params = read_params(request, 'params')
    X = build_features(df, params)
    cl = classifier.cl
    sklearn.metrics.roc_auc_score(label_test.label,
                                  tpot.predict_proba(X_test)[:,1])

if __name__ == "__main__":
    if os.environ.get('VCAP_SERVICES') is None: # running locally
        PORT = 8080
        DEBUG = True
    else:                                       # running on CF
        PORT = int(os.getenv("PORT"))
        DEBUG = False

    app.run(host='0.0.0.0', port=PORT, debug=DEBUG)
