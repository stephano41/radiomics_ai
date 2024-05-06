# Radiomics AI

This repository contains experimental code used for developing radiomics machine learning models. The terminal interface is powered by Hydra.

## Running experiments:
### Build the container
Build the container: `docker compose build`\
without docker compose: `docker build . -t radiomics_ai`

**Main Pipeline**\
Main pipeline is run by using the `main.py` script and specifying an experiment config\
With docker compose: `docker compose run app [command]`\
Without docker compose: `docker run --gpus all --shm-size=8gb -it -v "$(pwd)":/opt/project/ --env AUTORAD_RESULT_DIR=./outputs --env TZ=Australia/Adelaide --env ENABLE_AUTORAD_LOGGING=0 --env HYDRA_FULL_ERROR=1 --name "radiomics_ai_$(date +'%Y%m%d%H%M%S')" radiomics_ai:latest [command]`\
Replace `[command]` with the specific command you want to execute

At each run, a new run folder will be created in `outputs/experiment_name/YYYY-MM-DD-HH-mm-dd`\
mlflow runs will be logged in `outputs/models/`

### Experiments supported
**Wikisarcoma**
 - Handcrafted radiomics-only: `docker compose run app python main.py experiments=wiki_sarcoma`

**Meningioma**
 - Handcrafted radiomics-only: `docker compose run app python main.py experiments=meningioma`
 - handcrafted + deep learning radiomics: `docker compose run app python main.py experiments=meningioma_autoencoder`

### Special Pipelines:
The main pipeline used to run experiments by default automatically includes the evaluation and analysis pipeline at the end of the run, however, there may be cases for only running the evaluation and analysis.
<details>
<summary>Evaluation</summary>
<code>docker compose run app python main.py experiments=meningioma pipelines=evaluate_run</code><br>
Runs only the evaluation portion of the main pipeline, it runs the bootstrap component and generates confidence intervals for AUC ROC, speicifcity, sensitivity, positive predictive value and negative predictive value, in addition to the AUC ROC curve.
It accepts a `run_id` argument to analyse a specific run, or it'll analyse the last run of the experiment name in the config

</details>
<details>
<summary>Analysis</summary>
<code>docker compose run app python main.py experiments=meningioma pipelines=analysis
</code><br>
Runs only the analysis portion of the main pipeline, mainly for explaining and assessing model performance.
It accepts a `run_id` argument to analyze a specific run, or it'll analyze the last run of the experiment name in the config.
It calculates the Shapley values of the selected model in the run, and plots:

 - The global absolute SHAP bar plot
 - Breakdown of the bar plot according to feature class and imaging modalities
 - Scatter plot of the top features
 - Box and whisker plots of the chosen features against the outcome
If the bootstrap scores have been calculated, it'll additionally plot:
 - Calibration curve
 - Decision analysis curve
</details>
<details>
<summary>Compare2models pipeline</summary>
<code>docker compose run app python main.py experiments=meningioma pipelines=compare2models model1_run_id=??? model2_run_id=???
</code><br>
Requires specifying <code>model1_run_id</code> and <code>model2_run_id</code> to get the model and dataset artifacts
Uses the Combined 5x2 f test to evaluate the difference in performance between 2 models
</details>
<details>
<summary>Calculate sample size pipeline</summary>
<code>docker compose run app python main.py experiments=meningioma pipeline._target_=src.pipeline.get_sample_size +sample_sizes=[115,110,100,90,80,70,60,50,40,30,20] name=calculate_sample_size
</code><br>
Uses a post hoc subsampling method to calculate whether a sufficient sample size was used to train the model, returns a box and whisker plot

</details>


### Viewing experiment results
Results are stored in the hydra output folder and the artifacts folder of mlflow\
All results can be viewed via mlflow, whereas the detailed run configs would be seen in the hydra output folder\
To launch mlflow:\
`docker compose up mlflow`\
If running as dockerfile:\
`docker run --shm-size=1gb -it --rm -v "$(pwd)":/opt/project/ --env AUTORAD_RESULT_DIR=./outputs --env TZ=Australia/Adelaide --env ENABLE_AUTORAD_LOGGING=0 --env HYDRA_FULL_ERROR=1 -p 8000:8000 --name "radiomics_ai_$(date +'%Y%m%d%H%M%S')" radiomics_ai:latest mlflow ui --host=0.0.0.0 --port=8000 --backend-store-uri=./outputs/models`

### Test runs

Sometimes it's necessary to see if a new configuration  will run without any breaks in the code, as such the `test_dryrun` function in `tests/test_dryrun.py` can help quickly identify problems. \
`docker compose run app pytest tests/test_dryrun.py::test_dryrun`


### Hydra terminal run tips
 - For config parameters that don't exist, you'll need to add a plus, for example:`+variable_name=VALUE` translates to config.variable_name: VALUE
 - To pass a list of numbers in the terminal, don't use quotation marks around the list, for example: `sample_size="[10,20,30]"` translates to a list of strings whereas `sample_size=[10,20,30]` translates to a list of numbers
 - Adding a notes argument `"+notes='helloworld'"` allows you to log notes descriptive of the experiment you are running (remember double quotes the whole argument and single quotes the value)

 ## Config parameters
 Available classifiers:
- Random Forest
- SVM
- XGBoost
- Logistic Regression
- KNN
- MLP
- DecisionTreeClassifier
- Catboost

Available oversampling methods:
- SMOTE
- ADASYN
- BorderlineSMOTE
- SMOTETomek
- SMOTEENN

Available feature selection methods:
- anova
- lasso
- linear_svc
- tree
- sf
- rfe
- mrmr
- pca

Classifiers, oversampling methods and feature selection methods support arguments being passed in the config. For example: `{_method_: sf, direction: backward, n_jobs: 5, tol: 0.05}` the `_method_` keyword must be used to indicate the method used.
## Development setup ##

### Pycharm interpreter setup
Setup interpreter with pycharm using the docker compose interpreter setting, don't adjust any other run time settings 
in pycharm (may result in breaks)

### Visual Studio Code
The Dev Containers extension allows easy attachment to a running container, such as `docker compose up app`

### Jupyter notebook setup
run: `docker compose up jupyter`\
or without docker compose: `docker run --shm-size=1gb -it --rm -v "$(pwd)":/opt/project/ --env AUTORAD_RESULT_DIR=./outputs --env TZ=Australia/Adelaide --env ENABLE_AUTORAD_LOGGING=0 --env HYDRA_FULL_ERROR=1 --name "radiomics_ai_$(date +'%Y%m%d%H%M%S')" -p 8888:8888 radiomics_ai:latest jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --no-browser --NotebookApp.token=eabc82b4d609af51618a78b0e93f791ff106b626a0c07ab7`\
The url to the notebook would be `http://127.0.0.1:8888`. Make sure the working directory in the notebook is correct before working
### Pytest
To run pytests: `docker compose run app pytest`

