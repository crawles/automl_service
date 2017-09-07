# AutoML Service

Deploy automated machine learning (AutoML) as a service using `Flask`, for both pipeline training and pipeline serving. 

The framework implements a fully automated time series classification pipeline, automating both feature engineering and model selection and optimization using Python libraries, `TPOT` and `tsfresh`.


<p>
  <img src="https://github.com/crawles/Logos/blob/master/automl.gif?raw=true" width = 80%>
</p>


Resources:

- [TPOT](https://github.com/rhiever/tpot)– Automated feature preprocessing and model optimization tool
- [tsfresh](https://github.com/blue-yonder/tsfresh)– Automated time series feature engineering and selection
 
## Architecture

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

## Running the tests

Supply a user argument for the host.

```
# use local app
py.test --host http://0.0.0.0:8080
```

```
# use cloud-deployed app
py.test --host http://automl.cfapps.pez.pivotal.io
```

## Scaling the architecture

For production, I would suggest splitting training and serving into seperate applications. Also it would be best to use a shared cache such as Redis or Pivotal Cloud Cache to allow other applications and multiple instances of the pipeline to access the trained model. Here is a potential architecture.

<p>
  <img src="/img/cloud_architecture.png?raw=true" width = 55%>
</p>
A scalable model training and model serving architecture.

## Author

`Chris Rawles`


