import argparse

from logger import configure_logger
import os
import pickle

import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor


def parse_arguments():

    """Function to parse the arguments

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
        "processed_dataset_path",
        type=str,
        help="Dataset path",
        nargs="?",
        const=1,
        default="data/processed",
    )
    parser.add_argument(
        "Pickle_path",
        type=str,
        help="The path to save the pickle file",
        nargs="?",
        const=1,
        default="artifacts",
    )
    parser.add_argument(
        "logfile",
        type=str,
        help="Logging file output",
        nargs="?",
        const=1,
        default="logs/train_log.txt",
    )
    args = parser.parse_args()
    return args


def train_models():

    """Function to train the models.

    Parameters
    ----------
    None

    Returns
    -------
    bool
        True if successful, False otherwise.

    """

    args = parse_arguments()
    processed_dataset_path = args.processed_dataset_path
    Pickle_path = args.Pickle_path
    log_file_path = args.logfile
    logging = configure_logger(log_file=log_file_path, console=False, log_level="INFO")

    # strat_train_set = pd.read_csv(os.path.join(processed_dataset_path,'strat_train_set.csv'))
    # strat_test_set = pd.read_csv(os.path.join(processed_dataset_path,'strat_test_set.csv'))
    logging.info("Loading the housing labels data from csv")
    housing_labels = pd.read_csv(
        os.path.join(processed_dataset_path, "housing_labels.csv")
    )
    logging.info("Loading the housing prepared data from csv")
    housing_prepared = pd.read_csv(
        os.path.join(processed_dataset_path, "housing_prepared.csv")
    )
    logging.info("Building Linear Regression model")
    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)
    logging.info("Saving the Linear Regression model")
    with open(os.path.join(Pickle_path, "linear_reg_model.pkl"), "wb") as file1:
        pickle.dump(lin_reg, file1)

    housing_predictions = lin_reg.predict(housing_prepared)
    lin_mse = mean_squared_error(housing_labels, housing_predictions)
    lin_rmse = np.sqrt(lin_mse)
    lin_rmse

    lin_mae = mean_absolute_error(housing_labels, housing_predictions)
    lin_mae
    logging.info("Building DecisionTreeRegressor model")
    tree_reg = DecisionTreeRegressor(random_state=42)
    tree_reg.fit(housing_prepared, housing_labels.values.ravel())
    logging.info("Saving the DecisionTreeRegressor model")
    with open(os.path.join(Pickle_path, "DecisionTree_reg_model.pkl"), "wb") as file2:
        pickle.dump(tree_reg, file2)

    housing_predictions = tree_reg.predict(housing_prepared)
    tree_mse = mean_squared_error(housing_labels, housing_predictions)
    tree_rmse = np.sqrt(tree_mse)
    tree_rmse

    param_distribs = {
        "n_estimators": randint(low=1, high=200),
        "max_features": randint(low=1, high=8),
    }

    forest_reg = RandomForestRegressor(random_state=42)
    rnd_search = RandomizedSearchCV(
        forest_reg,
        param_distributions=param_distribs,
        n_iter=10,
        cv=5,
        scoring="neg_mean_squared_error",
        random_state=42,
    )
    rnd_search.fit(housing_prepared, housing_labels.values.ravel())
    cvres = rnd_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)

    param_grid = [
        # try 12 (3×4) combinations of hyperparameters
        {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
        # then try 6 (2×3) combinations with bootstrap set as False
        {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
    ]

    forest_reg = RandomForestRegressor(random_state=42)
    # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
    grid_search = GridSearchCV(
        forest_reg,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        return_train_score=True,
    )
    grid_search.fit(housing_prepared, housing_labels.values.ravel())

    grid_search.best_params_
    cvres = grid_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)

    feature_importances = grid_search.best_estimator_.feature_importances_
    sorted(zip(feature_importances, housing_prepared.columns), reverse=True)
    logging.info("Final model obtained using grid_search.best_estimator_")
    final_model = grid_search.best_estimator_

    logging.info("Saving the final model")
    with open(os.path.join(Pickle_path, "final_model.pkl"), "wb") as f:
        pickle.dump(final_model, f)
    return True


if __name__ == "__main__":
    train_models()
