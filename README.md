# AutoML Service

Deploy automated machine learning (AutoML) as a service using `Flask`, for both pipeline training and pipeline serving. 

The framework implements a fully automated time series classification pipeline, automating both feature engineering and model selection and optimization using Python libraries, `TPOT` and `tsfresh`. 

Check out the [blog post](https://content.pivotal.io/blog/automated-machine-learning-deploying-automl-to-the-cloud) for more info.

<p>
  <img src="https://github.com/crawles/Logos/blob/master/automl.gif?raw=true" width = 80%>
</p>

Resources:

- [TPOT](https://github.com/rhiever/tpot)– Automated feature preprocessing and model optimization tool
- [tsfresh](https://github.com/blue-yonder/tsfresh)– Automated time series feature engineering and selection
- [Flask](http://flask.pocoo.org/)– A web development microframework for Python
 
## Architecture

The application exposes both model training and model predictions with a RESTful API. For model training, input data and labels are sent via POST request, a pipeline is trained, and model predictions are accessible via a prediction route.

Pipelines are stored to a unique key, and thus, live predictions can be made on the same data using different feature construction and modeling pipelines.

<p>
  <img src="/img/architecture.png?raw=true" width = 55%>
</p>

An automated pipeline for time-series classification.

<p>
  <img src="/img/training.png?raw=true" width = 55%>
</p>
The model training logic is exposed as a REST endpoint. Raw, labeled training data is uploaded via a POST request and an optimal model is developed.
<p>
  <img src="/img/serving.png?raw=true" width = 55%>
</p>
Raw training data is uploaded via a POST request and a model prediction is returned.

## Using the app
View the [Jupyter Notebook](https://github.com/crawles/automl_service/blob/master/modelling_and_usage.ipynb) for an example.
### Deploying


```bash
# deploy locally
python automl_service.py
```

```bash
# deploy on cloud foundry
cf push
```
### Usage

Train a pipeline:

```python
train_url = 'http://0.0.0.0:8080/train_pipeline'
train_files = {'raw_data': open('data/data_train.json', 'rb'),
               'labels'  : open('data/label_train.json', 'rb'),
               'params'  : open('parameters/train_parameters_model2.yml', 'rb')}

# post request to train pipeline
r_train = requests.post(train_url, files=train_files)
result_df = json.loads(r_train.json())
```
returns:
```python
{'featureEngParams': {'default_fc_parameters': "['median', 'minimum', 'standard_deviation', 
                                                 'sum_values', 'variance', 'maximum', 
                                                 'length', 'mean']",
                      'impute_function': 'impute',
                      ...},
 'mean_cv_accuracy': 0.865,
 'mean_cv_roc_auc': 0.932,
 'modelId': 1,
 'modelType': "Pipeline(steps=[('stackingestimator', StackingEstimator(estimator=LinearSVC(...))),
                               ('logisticregression', LogisticRegressionClassifier(solver='liblinear',...))])"
 'trainShape': [1647, 8],
 'trainTime': 1.953}
 ```

Serve pipeline predictions:
```python
serve_url = 'http://0.0.0.0:8080/serve_prediction'
test_files = {'raw_data': open('data/data_test.json', 'rb'),
              'params' : open('parameters/test_parameters_model2.yml', 'rb')}

# post request to serve predictions from trained pipeline
r_test  = requests.post(serve_url, files=test_files)
result = pd.read_json(r_test.json()).set_index('id')
```

| example_id    | prediction    |
| ------------- | ------------- |
| 1             | 0.853         |
| 2             | 0.991         |
| 3             | 0.060         |
| 4             | 0.995         |
| 5             | 0.003         |
| ...           | ...           |

View all trained models:

```python
r = requests.get('http://0.0.0.0:8080/models')
pipelines = json.loads(r.json())
```

```python
{'1':
    {'mean_cv_accuracy': 0.873,
     'modelType': "RandomForestClassifier(...),
     ...},
 '2':
    {'mean_cv_accuracy': 0.895,
     'modelType': "GradientBoostingClassifier(...),
     ...},
 '3':
    {'mean_cv_accuracy': 0.859,
     'modelType': "LogisticRegressionClassifier(...),
     ...},
...}
```

## Running the tests

Supply a user argument for the host.

```bash
# use local app
py.test --host http://0.0.0.0:8080
```

```bash
# use cloud-deployed app
py.test --host http://ROUTE-HERE
```

## Scaling the architecture

For production, I would suggest splitting training and serving into seperate applications, and incorporating a fascade API. Also it would be best to use a shared cache such as Redis or Pivotal Cloud Cache to allow other applications and multiple instances of the pipeline to access the trained model. Here is a potential architecture.

<p>
  <img src="/img/cloud_architecture.png?raw=true" width = 55%>
</p>
A scalable model training and model serving architecture.

## Author

`Chris Rawles`

