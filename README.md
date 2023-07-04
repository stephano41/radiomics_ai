# MRI_sarcoma_AI
sarcoma MRI images analysis

## Dockerfile:
To build the image: `docker build -t mri_sarcoma_ai -f .\Dockerfile .` 
To access terminal of image: `docker run -it --name mri_sarcoma_ai mri_sarcoma_ai:latest`
To access mlflow server on docker: `docker run -p 5000:5000 -it --name mri_sarcoma_ai mri_sarcoma_ai:latest`