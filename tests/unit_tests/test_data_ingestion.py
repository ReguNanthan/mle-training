import os

import pandas as pd

from HousePricePrediction import ingest_data


def test_arguments():
    args = ingest_data.parse_arguments()
    assert os.path.exists(args.rawdatasetpath)
    assert os.path.exists(args.processeddatasetpath)
    assert os.path.exists(args.logfile)  # "logs/ingest_data_log.txt")


def test_fetch_dataset():

    DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
    HOUSING_PATH = "data/raw"
    print("housing_path :", HOUSING_PATH)
    HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
    ingest_data.fetch_housing_data(HOUSING_URL, HOUSING_PATH)
    assert os.path.exists("data/raw/housing.csv")


def test_load_dataset():
    raw_data_path = "data/raw/"
    df = ingest_data.load_housing_data(raw_data_path)
    assert isinstance(df, pd.DataFrame)


def test_transform_data():
    ingest_data.transform_data()
    assert os.path.exists("data/processed/housing_labels.csv")
    assert os.path.exists("data/processed/housing_prepared.csv")
    assert os.path.exists("data/processed/xtest_prepared.csv")
    assert os.path.exists("data/processed/y_test_prepared.csv")
