import json
import time

import pandas as pd
import sklearn
import tpot
import yaml
import werkzeug

from utilities import build_features, cross_validate, read_file, read_params,\
    train_model

from flask_restful import reqparse, Resource

class Model(Resource):

    def __init__(self, model_factory):
        self.model_factory = model_factory
        self.parser = reqparse.RequestParser()
        self.parser.add_argument('pipeline_id')
    
    def get(self):
        args = self.parser.parse_args()
        pipeline_id = args['pipeline_id']
        pipeline = self.model_factory[pipeline_id]
        result = dict()
        result['model'] = pipeline['stats']
        result['extract_features'] = pipeline['extract_features']
        return json.dumps(result)

class Models(Resource):

    def __init__(self, model_factory):
        self.model_factory = model_factory
    
    def get(self):
        model_ids = self.model_factory.pipelines.keys()
        result = dict()
        for m in model_ids:
            result[m] = self.model_factory[m]['stats']
        return json.dumps(result)



class Train(Resource):

    def __init__(self, model_factory):
        self.model_factory = model_factory
        self.parser = reqparse.RequestParser()
        self.parser.add_argument('params', type=werkzeug.FileStorage,
            location='files')
        self.parser.add_argument('raw_data', type=werkzeug.FileStorage, 
            location='files')
        self.parser.add_argument('labels', type=werkzeug.FileStorage, 
            location='files')

    def post(self):
        start_time = time.time()
        args = self.parser.parse_args()

        # read data
        params = read_params(args['params'].stream)
        df = read_file(args['raw_data'].stream.read())
        y_train = read_file(args['labels'].stream.read())

        # build features
        X_train = build_features(df, params)
        y_train = y_train.set_index('example_id')
        y_train = y_train.loc[X_train.index]

        # train model
        cl = train_model(X_train, y_train.label, params)
        self.model_factory.add_pipeline(cl, params)
        if isinstance(cl, tpot.TPOTClassifier):
            final_classifier = cl.fitted_pipeline_
            evaluated_indivs = cl.evaluated_individuals_
        else:
            final_classifier = cl
            evaluated_indivs = None
        model_type = str(final_classifier)
        mean_accuracy, mean_roc_auc = cross_validate(final_classifier,
                                                     X_train,
                                                     y_train.label)

        # format feat_eng_params
        feat_eng_params = params['extract_features'].copy()
        for k in feat_eng_params.keys():
            if k =='default_fc_parameters':  # shows calculations like min, mean, etc.
                feat_eng_params[k] = str(feat_eng_params[k].keys())
            elif k =='impute_function':
                feat_eng_params[k] = str(feat_eng_params[k].__name__)
            else:
                feat_eng_params[k] = str(feat_eng_params[k])

#        for k in feat_eng_params:
#            feat_eng_params[k] = str(feat_eng_params[k])
        result = {'trainTime': time.time()-start_time, 
                  'trainShape': X_train.shape,
                  'modelType': model_type,
                  'featureEngParams': feat_eng_params,
                  'modelId': params['pipeline_id'],
                  'mean_cv_accuracy' : mean_accuracy,
                  'mean_cv_roc_auc'  : mean_roc_auc,
                  'evaluated_models' : evaluated_indivs}
        self.model_factory[params['pipeline_id']]['stats'] = result
        return json.dumps(result)


class ServePrediction(Resource):

    def __init__(self, model_factory):
        self.model_factory = model_factory
        self.parser = reqparse.RequestParser()
        self.parser.add_argument('params', type=werkzeug.FileStorage,
            location='files')
        self.parser.add_argument('raw_data', type=werkzeug.FileStorage, 
            location='files')

    def post(self):
        args = self.parser.parse_args()
        params = read_params(args['params'].stream)
        df = read_file(args['raw_data'].stream.read())
        result = self.model_factory.use_pipeline(df, params['pipeline_id'])
        return result.reset_index().to_json()

