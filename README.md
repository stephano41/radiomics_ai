# MRI_sarcoma_AI
sarcoma MRI images analysis

## Docker Compose:
Build the container: `docker compose build`\
Launch mlflow server to view runs: `docker compose up`\
run wiki sarcoma experiment, in docker terminal: `python main.py`\
run meningioma experiment, in docker terminal: `python main.py "experiments=meningioma"`\


setup interpreter with pycharm using the docker compose interpreter setting, don't adjust any other run time settings 
in pycharm