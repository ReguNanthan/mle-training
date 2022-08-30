import argparse
from HousePricePrediction.logger import configure_logger
import os
import pickle

import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor

import mlflow
import mlflow.sklearn

from HousePricePrediction import score


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


def model_linear_regression(housing_prepared, housing_labels):
    """Function that runs the ML model

    Parameters
    ----------
    housing_prepared  : pandas Dataframe
        Dataframe that is needed for training

    housing_labels : pandas Dataframe
        Dataframe that is needed for training


    Returns
    --------
    object
        Returns the model that is trained

    """

    logging.info("Building Linear Regression model")
    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)

    mlflow.sklearn.log_model(lin_reg, "linear_reg_model")

    logging.info("Saving the Linear Regression model")
    with open(os.path.join(Pickle_path, "linear_reg_model.pkl"), "wb") as file1:
        pickle.dump(lin_reg, file1)

    housing_predictions = lin_reg.predict(housing_prepared)
    # lin_mse = mean_squared_error(housing_labels, housing_predictions)
    # lin_rmse = np.sqrt(lin_mse)

    # lin_r2 = r2_score(housing_labels, housing_labels)

    # lin_mae = mean_absolute_error(housing_labels, housing_predictions)

    return lin_reg


def model_decision_tree(housing_prepared, housing_labels):

    """Function that runs the ML model

    Parameters
    ----------
    housing_prepared  : pandas Dataframe
        Dataframe that is needed for training

    housing_labels : pandas Dataframe
        Dataframe that is needed for training


    Returns
    --------
    object
        Returns the model that is trained

    """
    logging.info("Building DecisionTreeRegressor model")
    tree_reg = DecisionTreeRegressor(random_state=42)
    tree_reg.fit(housing_prepared, housing_labels.values.ravel())

    mlflow.sklearn.log_model(tree_reg, "DecisionTree_reg_model")

    logging.info("Saving the DecisionTreeRegressor model")
    with open(os.path.join(Pickle_path, "DecisionTree_reg_model.pkl"), "wb") as file2:
        pickle.dump(tree_reg, file2)

    # housing_predictions = tree_reg.predict(housing_prepared)
    # tree_mse = mean_squared_error(housing_labels, housing_predictions)
    # tree_rmse = np.sqrt(tree_mse)
    # tree_rmse

    # tree_r2 = r2_score(housing_labels, housing_labels)

    # tree_mae = mean_absolute_error(housing_labels, housing_predictions)
    # tree_mae

    return tree_reg


def model_RandomSearch_RandomForest(housing_prepared, housing_labels):

    """Function that runs the ML model

    Parameters
    ----------
    housing_prepared  : pandas Dataframe
        Dataframe that is needed for training

    housing_labels : pandas Dataframe
        Dataframe that is needed for training


    Returns
    --------
    object
        Returns the model that is trained

    """
    # Building RandomizedSearchCV RandomForestRegressor

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
        print("RandomSearch_RandomForest_Model", np.sqrt(-mean_score), params)

    rnd_search_best_estimator = rnd_search.best_estimator_

    mlflow.sklearn.log_model(
        rnd_search_best_estimator, "RandomSearch_RandomForest_Model"
    )

    logging.info("Saving the RandomSearch_RandomForest_Model")
    with open(
        os.path.join(Pickle_path, "RandomSearch_RandomForest_Model.pkl"), "wb"
    ) as f:
        pickle.dump(rnd_search_best_estimator, f)

    return rnd_search_best_estimator


def model_GridSearch_RandomForest(housing_prepared, housing_labels):

    """Function that runs the ML model

    Parameters
    ----------
    housing_prepared  : pandas Dataframe
        Dataframe that is needed for training

    housing_labels : pandas Dataframe
        Dataframe that is needed for training


    Returns
    --------
    object
        Returns the model that is trained

    """

    # Building Grid Search CV

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
        print("Grid Search CV", np.sqrt(-mean_score), params)

    feature_importances = grid_search.best_estimator_.feature_importances_
    sorted(zip(feature_importances, housing_prepared.columns), reverse=True)
    logging.info("Final model obtained using grid_search.best_estimator_")
    final_model = grid_search.best_estimator_
    print("best estimator :", grid_search.best_estimator_)

    mlflow.sklearn.log_model(final_model, "Final_Model")

    logging.info("Saving the final model")
    with open(os.path.join(Pickle_path, "final_model.pkl"), "wb") as f:
        pickle.dump(final_model, f)

    return final_model


def load_datasets():

    """Function to load the datasets

    Parameters
    ----------
    None

    Returns
    -------
    housing_labels
        Dataset that contains the housing prices needed for training.
    housing_prepared
        Dataset that contains the housing details needed for training.

    """

    args = parse_arguments()
    processed_dataset_path = args.processed_dataset_path
    Pickle_path = args.Pickle_path
    log_file_path = args.logfile
    logging = configure_logger(log_file=log_file_path, console=False, log_level="INFO")

    logging.info("Loading the housing labels data from csv")
    housing_labels = pd.read_csv(
        os.path.join(processed_dataset_path, "housing_labels.csv")
    )
    logging.info("Loading the housing prepared data from csv")
    housing_prepared = pd.read_csv(
        os.path.join(processed_dataset_path, "housing_prepared.csv")
    )

    # logging.info("Downloading the Xtest and Ytest from processed path folder")

    # X_test_prepared = pd.read_csv(os.path.join(processed_path, "xtest_prepared.csv"))
    # y_test = pd.read_csv(os.path.join(processed_path, "y_test_prepared.csv"))

    return housing_prepared, housing_labels


def main(model_name):

    """Function that runs the corresponding model and also calculate the scores

    Parameters
    ----------
    model_name  : str
        Name of the model that needs to be run.


    Returns
    --------
    bool
        True if successful, False otherwise.

    """

    housing_prepared, housing_labels = load_datasets()
    if model_name == "Linear Regressor":
        model = model_linear_regression(housing_prepared, housing_labels)
        score.predict_score(model_name, model)
    elif model_name == "Decision Tree Regressor":
        model = model_decision_tree(housing_prepared, housing_labels)
        score.predict_score(model_name, model)
    elif model_name == "RandomSearch RandomForest Regressor":
        model = model_RandomSearch_RandomForest(housing_prepared, housing_labels)
        score.predict_score(model_name, model)
    else:
        model = model_GridSearch_RandomForest(housing_prepared, housing_labels)
        model_name = "GridSearchCV RandomForest Regressor"
        score.predict_score(model_name, model)

    return True


if __name__ == "__main__":
    args = parse_arguments()
    processed_dataset_path = args.processed_dataset_path
    Pickle_path = args.Pickle_path
    log_file_path = args.logfile
    logging = configure_logger(log_file=log_file_path, console=False, log_level="INFO")
    main("")
else:
    args = parse_arguments()
    processed_dataset_path = args.processed_dataset_path
    Pickle_path = args.Pickle_path
    log_file_path = args.logfile
    logging = configure_logger(log_file=log_file_path, console=False, log_level="INFO")
    # main("")
