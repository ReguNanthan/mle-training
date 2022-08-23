import argparse
import pickle
import os
import pandas as pd
import numpy as np

from logger import configure_logger

from sklearn.metrics import mean_squared_error


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


def predict():

    """
    Function to predict the output and score

    Parameters
    ----------
    None

    Returns
    -------
    bool
        True if successful, False otherwise.

    """

    args = parse_arguments()

    model_path = args.model_folder

    processed_path = args.processed_path

    log_file_path = args.log_file
    logging = configure_logger(log_file=log_file_path, console=False, log_level="INFO")

    logging.info("Downloading the Xtest and Ytest from processed path folder")

    X_test_prepared = pd.read_csv(os.path.join(processed_path, "xtest_prepared.csv"))
    y_test = pd.read_csv(os.path.join(processed_path, "y_test_prepared.csv"))

    logging.info("Unpickling the final model")

    with open(os.path.join(model_path, "final_model.pkl"), "rb") as file_handle:
        final_model = pickle.load(file_handle)

    logging.info("Prediction using Final Model")

    final_predictions = final_model.predict(X_test_prepared)
    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)
    print("final model rmse", final_rmse)

    return True


if __name__ == "__main__":
    predict()
