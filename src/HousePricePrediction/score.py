import argparse
import pickle
import os
import pandas as pd
import numpy as np

from logger import configure_logger

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import mlflow


def parse_arguments():
    """
    Function to parse the arguments

    Parameters
    ----------
    None

    Returns
    -------
    Args
        Returns the arguments that are added in the argument parser

    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_folder",
        type=str,
        help="path where the model is saved",
        nargs="?",
        const=1,
        default="artifacts",
    )
    parser.add_argument(
        "processed_path",
        type=str,
        help="path where the processed datasets are stored",
        nargs="?",
        const=1,
        default="data/processed",
    )
    parser.add_argument(
        "log_file",
        type=str,
        help="Logging file output",
        nargs="?",
        const=1,
        default="logs/score_log.txt",
    )

    args = parser.parse_args()
    return args


def predict_score(model_name, model):

    """Function that predict and return the scores of a ML model.

    Parameters
    ----------
    model_name  : str
        Name of the model that needs to evaluated.

    model : object
        model object


    Returns
    --------
    bool
        Returns True if successful, otherwise False.

    """

    args = parse_arguments()

    model_path = args.model_folder

    processed_path = args.processed_path

    log_file_path = args.log_file
    logging = configure_logger(log_file=log_file_path, console=False, log_level="INFO")

    logging.info("Downloading the Xtest and Ytest from processed path folder")

    X_test_prepared = pd.read_csv(os.path.join(processed_path, "xtest_prepared.csv"))
    y_test = pd.read_csv(os.path.join(processed_path, "y_test_prepared.csv"))

    logging.info(f"Unpickling the {model_name} model")

    if model_name == "Linear Regressor":
        with open(
            os.path.join(model_path, "linear_reg_model.pkl"), "rb"
        ) as file_handle:
            model = pickle.load(file_handle)
    elif model_name == "Decision Tree Regressor":
        with open(
            os.path.join(model_path, "DecisionTree_reg_model.pkl"), "rb"
        ) as file_handle:
            model = pickle.load(file_handle)
    elif model_name == "RandomSearch RandomForest Regressor":
        with open(
            os.path.join(model_path, "RandomSearch_RandomForest_Model.pkl"), "rb"
        ) as file_handle:
            model = pickle.load(file_handle)
    else:
        with open(os.path.join(model_path, "final_model.pkl"), "rb") as file_handle:
            model = pickle.load(file_handle)
            model_name = "GridSearchCV RandomForest Regressor"

    logging.info(f"Prediction using {model_name} ")

    housing_predictions = model.predict(X_test_prepared)
    r2 = r2_score(y_test, housing_predictions)
    mae = mean_absolute_error(y_test, housing_predictions)
    mse = mean_squared_error(y_test, housing_predictions)

    rmse = np.sqrt(mse)
    print(f"{model_name} rmse", rmse)

    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)

    return True


if __name__ == "__main__":
    predict_score("RandomSearch RandomForest Regressor", "")
    # main()
