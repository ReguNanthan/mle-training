import argparse
import pickle
import os
import pandas as pd
import numpy as np

from logger import configure_logger

from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer


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

    strat_test_set = pd.read_csv(os.path.join(processed_path, "strat_test_set.csv"))

    strat_test_set.drop("income_cat", axis=1, inplace=True)

    # strat_test_set.drop('Unnamed: 0', axis=1, inplace=True)
    # print(strat_test_set.columns)
    # Loading model to compare the results
    # final_model = pickle.load(open(os.path.join(model_path,'final_model.pkl'),'rb'))

    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()

    imputer = SimpleImputer(strategy="median")

    X_test_num = X_test.drop("ocean_proximity", axis=1)
    imputer.fit(X_test_num)
    X_test_prepared = imputer.transform(X_test_num)
    X_test_prepared = pd.DataFrame(
        X_test_prepared, columns=X_test_num.columns, index=X_test.index
    )
    X_test_prepared["rooms_per_household"] = (
        X_test_prepared["total_rooms"] / X_test_prepared["households"]
    )
    X_test_prepared["bedrooms_per_room"] = (
        X_test_prepared["total_bedrooms"] / X_test_prepared["total_rooms"]
    )
    X_test_prepared["population_per_household"] = (
        X_test_prepared["population"] / X_test_prepared["households"]
    )

    X_test_cat = X_test[["ocean_proximity"]]
    X_test_prepared = X_test_prepared.join(pd.get_dummies(X_test_cat, drop_first=True))

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
