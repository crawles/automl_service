import json

import werkzeug

from flask_restful import reqparse, Resource

class Models(Resource):

    def get(self):
        return json.dumps({1: 'model1', 2: 'model2', 3: 'model3'})

class Train(Resource):

    def __init__(self):
        self.parser = reqparse.RequestParser()
        self.parser.add_argument('params', type=werkzeug.FileStorage,
                location='files')

    def post(self):
        args = self.parser.parse_args()
        print 111
        print args['params'].stream
        print 111
        return json.dumps(args)
