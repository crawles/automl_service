# AutoML Service

Deploy an automated machine learning (AutoML) pipeline as a service using Flask, for both pipeline training and pipeline serving. 

The framework implements a fully automated time series classification pipeline, automating both feature engineering and model selection and optimization using Python libraries, TPOT and tsfresh.

![Alt text](/img/architecture.png?raw=true "architecture")

Resources:

- [TPOT](https://github.com/rhiever/tpot)– Automated feature preprocessing and model optimization tool
- [tsfresh](https://github.com/blue-yonder/tsfresh)– Automated time series feature engineering and selection
 
## Training and serving a model

![Alt text](/img/training.png?raw=true "training")
![Alt text](/img/serving.png?raw=true "serving")

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

![Alt text](/img/cloud_architecture.png?raw=true "cloud_architecture")

## Author

`Chris Rawles`
