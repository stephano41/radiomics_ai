# MRI_sarcoma_AI
sarcoma MRI images analysis\
Also contains data for wiki sarcoma, meningioma

## Docker Compose:
Build the container: `docker compose build`\
Launch mlflow server to view runs: `docker compose up app`\
run wiki sarcoma experiment, in docker app container terminal: `python main.py "experiments=wiki_sarcoma"`\
run meningioma experiment, in docker app container terminal: `python main.py "experiments=meningioma"`

## Pycharm interpreter setup
setup interpreter with pycharm using the docker compose interpreter setting, don't adjust any other run time settings 
in pycharm (may result in breaks)
### Jupyter notebook setup
run: `docker compose up jupyter`\
copy the generated token in the outputs, go to jupyter notebook and paste url and token (http://127.0.0.1:8888?token={token})
make sure the working directory in the notebook is correct before working
### Pytest
To run pytests: `docker compose run app pytest`