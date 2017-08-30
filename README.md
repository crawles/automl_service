# AutoML Service

Deploy an automated machine learning (AutoML) pipeline as a service using `Flask`, for both pipeline training and pipeline serving. 

The framework implements a fully automated time series classification pipeline, automating both feature engineering and model selection and optimization using Python libraries, `TPOT` and `tsfresh`.

<p>
  <img src="/img/architecture.png?raw=true" width = 55%>
</p>

Resources:

- [TPOT](https://github.com/rhiever/tpot)– Automated feature preprocessing and model optimization tool
- [tsfresh](https://github.com/blue-yonder/tsfresh)– Automated time series feature engineering and selection
 
## Training and serving a model

<p>
  <img src="/img/training.png?raw=true" width = 55%>
</p>
<p>
  <img src="/img/serving.png?raw=true" width = 55%>
</p>

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

## Author

`Chris Rawles`

