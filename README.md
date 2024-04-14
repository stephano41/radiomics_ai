# MRI_sarcoma_AI
sarcoma MRI images analysis\
Also contains data for wiki sarcoma, meningioma

## Running experiments:
### Build the container
Build the container: `docker compose build`\
Launch mlflow server to view runs: `docker compose up app`\
Main pipeliine is run by using the `main.py` script and specifying an experiment config\
You can also run only the bootstrap portion of the pipeline to evaluate a model by specifying `pipelines=evaluate_run`

### Experiments supported
**Wikisarcoma**
 - radiomics: `docker compose run app python main.py experiments=wiki_sarcoma`

**Meningioma**
 - radiomics only: `docker compose run app python main.py experiments=meningioma`
 - radiomics + deep learning autoencoder features: `docker compose run app python main.py experiments=meningioma_autoencoder`

### Special Pipelines:
<details>
<summary>Evaluation only</summary>
<code>docker compose run app python main.py experiments=meningioma pipeline._target_=src.pipeline.evaluate_run</code><br>
Accepts a `run_id` argument to analyse a specific run, or it'll analyse the last run of the experiment name in the config
</details>
<details>
<summary>Analysis pipeline</summary>
<code>docker compose run app python main.py experiments=meningioma pipeline._target_=src.pipeline.run_analysis
</code><br>
Accepts a run_id argument to analyze a specific run, or it'll analyze the last run of the experiment name in the config.
</details>
<details>
<summary>Compare2models pipeline</summary>
<code>docker compose run app python main.py experiments=meningioma pipelines=compare2models model1_run_id=??? model2_run_id=???
</code><br>
Requires specifying `model1_run_id` and `model2_run_id` to get the model and dataset artifacts
</details>

### Viewing experiment results
Results are stored between the hydra output folder and the artifacts folder of mlflow\
All results can be viewed via mlflow, whereas the detailed run configs would be seen in the hydra output folder\
To launch mlflow:\
`docker compose up app`\
If running as dockerfile:\
`sudo docker run --gpus all --shm-size=1gb -it --rm -v "$(pwd)":/opt/project/ --env AUTORAD_RESULT_DIR=./outputs --env TZ=Australia/Adelaide --env ENABLE_AUTORAD_LOGGING=0 --env HYDRA_FULL_ERROR=1 -p 8000:8000 steven_container:latest mlflow ui --host=0.0.0.0 --port=8000 --backend-store-uri=./outputs/models`

### Test runs

Sometimes it's necessary to see if the new run will run without any breaks in the code.
 With any experiment with autoencoder: `docker compose run app python main.py experiments={ANY_EXPERIMENT} "autoencoder=dummy_vae" "bootstrap.iters=5" "optimizer.n_trials=5 name=test_run"`

### Running as dockerfile:
To build the image:\
`sudo docker build . -t steven_container`\
To use the image:\
`sudo docker run --gpus all --shm-size=1gb -it -d -v "$(pwd)":/opt/project/ --env AUTORAD_RESULT_DIR=./outputs --env TZ=Australia/Adelaide --env ENABLE_AUTORAD_LOGGING=0 --env HYDRA_FULL_ERROR=1 --name "steven_container_$(date +'%Y%m%d%H%M%S')" steven_container:latest python main.py`\
Replace python main.py with whatever commands

### Hydra terminal run tips
 - For config parameters that don't exist, you'll need to add a plus, for example:`+variable_name=VALUE` translates to config.variable_name: VALUE
 - To pass a list of numbers in the terminal, don't use quotation marks around the list, for example: `sample_size="[10,20,30]"` translates to a list of strings whereas `sample_size=[10,20,30]` translates to a list of numbers
 - Adding a notes argument `"+notes='helloworld'"` allows you to log notes descriptive of the experiment you are running (remember double quotes the whole thing and single quotes the argument)

 ## Config parameters
 Available classifiers:
- Random Forest
- SVM
- XGBoost
- Logistic Regression
- KNN
- MLP
- DecisionTreeClassifier

Available oversampling methods:
- SMOTE
- ADASYN
- BorderlineSMOTE

Available feature selection methods:
- anova
- lasso
- linear_svc
- tree
- sf
- rfe
- mrmr
- pca

Oversampling methods and feature selection methods can support passing arguments. For example: `{_method_: sf, direction: backward, n_jobs: 5, tol: 0.05}` the `_method_` keyword must be used to indicate the method used.
## Development setup ##

### Pycharm interpreter setup
Setup interpreter with pycharm using the docker compose interpreter setting, don't adjust any other run time settings 
in pycharm (may result in breaks)
### Jupyter notebook setup
run: `docker compose up jupyter`\
copy the generated token in the outputs, go to jupyter notebook and paste url and token (http://127.0.0.1:8888?token={token})
make sure the working directory in the notebook is correct before working
### Pytest
To run pytests: `docker compose run app pytest`

