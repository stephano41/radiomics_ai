# MRI_sarcoma_AI
sarcoma MRI images analysis\
Also contains data for wiki sarcoma, meningioma

## Docker Compose:
Build the container: `docker compose build`\
Launch mlflow server to view runs: `docker compose up app`\
Main pipeliine is run by using the `main.py` script and specifying an experiment config\
You can also run only the bootstrap portion of the pipeline to evaluate a model by specifying `pipelines=evaluate_last`
### Running experiments
**Wikisarcoma**
 - radiomics: `docker compose run app python main.py experiments=wiki_sarcoma`

**Meningioma**
 - radiomics only: `docker compose run app python main.py experiments=meningioma`
 - radiomics + deep learning autoencoder features: `docker compose run app python main.py experiments=meningioma_autoencoder`

### Running Analysis pipeline
 `docker compose run app python main.py experiments=meningioma pipeline._target_=src.pipeline.run_analysis`

### Running evaluation pipeline only
 `docker compose run app python main.py experiments=meningioma pipeline._target_=src.pipeline.evaluate_last`

For running the analysis and evaluation pipeline, specifying a `run_id` argument will result in analysis/evaluation of that run_id, otherwise it'll default to the last run from the experiment name (`config.name`)

Running as dockerfile:
`sudo docker build . -t steven_container`
`sudo docker run --gpus all --shm-size=1gb -it -d -v "$(pwd)":/opt/project/ --env AUTORAD_RESULT_DIR=./outputs --env TZ=Australia/Adelaide --env ENABLE_AUTORAD_LOGGING=0 --env HYDRA_FULL_ERROR=1 -p 8000:8000 steven_container:latest python main.py`

**Test runs**
 - with any experiment with autoencoder: `docker compose run app python main.py experiments={ANY_EXPERIMENT} "autoencoder=dummy_vae" "bootstrap.iters=5" "optimizer.n_trials=5"`

**hydra terminal run tips**
 - For config parameters that don't exist, you'll need to add a plus, for example:`+variable_name=VALUE` translates to config.variable_name: VALUE
 - To pass a list of numbers in the terminal, don't use quotation marks around the list, for example: `sample_size="[10,20,30]"` translates to a list of strings whereas `sample_size=[10,20,30]` translates to a list of numbers

## Pycharm interpreter setup
setup interpreter with pycharm using the docker compose interpreter setting, don't adjust any other run time settings 
in pycharm (may result in breaks)
### Jupyter notebook setup
run: `docker compose up jupyter`\
copy the generated token in the outputs, go to jupyter notebook and paste url and token (http://127.0.0.1:8888?token={token})
make sure the working directory in the notebook is correct before working
### Pytest
To run pytests: `docker compose run app pytest`

