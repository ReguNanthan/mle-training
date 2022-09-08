FROM continuumio/miniconda3:latest

WORKDIR /usr/housepriceprediction

COPY ./ ./

RUN conda env create -f env.yml
SHELL ["conda", "run", "-n", "mle-dev", "/bin/bash", "-c"]
RUN pip install HousePricePrediction-0.5.1-py3-none-any.whl
RUN mkdir artifacts
RUN mkdir data
RUN mkdir data/processed
RUN mkdir logs
#RUN mlflow server --backend-store-uri mlruns/  --default-artifact-root mlruns/ --host 0.0.0.0 --port 9000

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "mle-dev", "python", "mlflow_run.py"]
