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
 - radiomics + deep learning autoencoder features: `docker comopose run app python main.py experiments=meningioma_autoencoder`

**Test runs**
 - with any experiment with autoencoder: `docker compose run app python main.py experiments={ANY_EXPERIMENT} "autoencoder=dummy_vae" "bootstrap.iters=5" "optimizer.n_trials=5"`

## Pycharm interpreter setup
setup interpreter with pycharm using the docker compose interpreter setting, don't adjust any other run time settings 
in pycharm (may result in breaks)
### Jupyter notebook setup
run: `docker compose up jupyter`\
copy the generated token in the outputs, go to jupyter notebook and paste url and token (http://127.0.0.1:8888?token={token})
make sure the working directory in the notebook is correct before working
### Pytest
To run pytests: `docker compose run app pytest`
