# -*- coding: utf-8 -*-
"""
automl_service.py
~~~~~~~~~~~~~~~~~

App implements an automl pipeline.

"""

import os
import json

from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import requests
import sklearn

import resources

from flask_restful import reqparse, abort, Api, Resource
from utilities import build_features, read_file, read_params,\
        train_model, ModelFactory


app = Flask(__name__)
api = Api(app)

model_factory = ModelFactory()


api.add_resource(resources.Train, '/train_model',
    resource_class_kwargs={'model_factory': model_factory})
api.add_resource(resources.ServePrediction, '/serve_prediction',
    resource_class_kwargs={'model_factory': model_factory})
api.add_resource(resources.Models, '/models',
    resource_class_kwargs={'model_factory': model_factory})



#@app.route('/train_model', methods=['POST'])
#def train_api():
#    
#    df = read_file(request, 'raw_data')
#    params = read_params(request, 'params')
#    X_train = build_features(df, params)
#    y_train = read_file(request, 'labels')
#    y_train = y_train.set_index('example_id')
#    y_train = y_train.loc[X_train.index]
#
#    cl = train_model(X_train, y_train.label, params)
#    classifier.cl = cl
#    print sklearn.metrics.roc_auc_score(y_train.label,
#                                        cl.predict_proba(X_train)[:,1])
#    return str(cl)


@app.route('/serve_pred', methods=['POST'])
def serve_api():
    df = read_file(request, 'raw_data')
    params = read_params(request, 'params')
    X = build_features(df, params)
    scores = classifier.cl.predict_proba(X)[:,1]
    result = pd.DataFrame(scores,
                          columns=['score'],
                          index=X.index)
    return result.to_json()


if __name__ == "__main__":
    if os.environ.get('VCAP_SERVICES') is None: # running locally
        PORT = 8080
        DEBUG = True
    else:                                       # running on CF
        PORT = int(os.getenv("PORT"))
        DEBUG = False

    app.run(host='0.0.0.0', port=PORT, debug=DEBUG)
