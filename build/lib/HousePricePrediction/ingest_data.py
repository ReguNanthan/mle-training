import os
import tarfile

import numpy as np
from six.moves import urllib


import argparse

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.impute import SimpleImputer

from logger import configure_logger


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
        "rawdatasetpath",
        type=str,
        help="path to store the raw housing dataset",
        nargs="?",
        const=1,
        default="data/raw",
    )
    parser.add_argument(
        "processeddatasetpath",
        type=str,
        help="path to store the train and val dataset",
        nargs="?",
        const=1,
        default="data/processed",
    )

    parser.add_argument(
        "logfile",
        type=str,
        help="Logging file output",
        nargs="?",
        const=1,
        default="logs/ingest_data_log.txt",
    )

    return parser.parse_args()


def fetch_housing_data(housing_url, housing_path):
    """Fetch the Raw Datasets.

    Parameters
    ----------
    housing_url  : str
        Path to download housing url

    housing_path : str
        Path to store housing dataset

    Returns
    --------
    bool
        True if successful, False otherwise.

    """
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
    return True


def load_housing_data(housing_path):

    """Load the raw dataset.

    Parameters
    ----------
    housing_path  : str
        Path to fetch the housing dataset

    Returns
    --------
    Object
        Returns a pandas object with housing dataset

    """

    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def income_cat_proportions(data):

    """Details on income categories

    Parameters
    ----------
    data  : pandas Dataframe
        Data to be analyzed

    Returns
    --------
    int
        Returns income category count

    """

    return data["income_cat"].value_counts() / len(data)


def transform_data():

    """Transform the raw datasets to train and test datasets

    Parameters
    ----------
    None

    Returns
    --------
    bool
        True if successful, False otherwise.

    This function stores the datasets in the processed folder


    """

    DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
    args = parse_arguments()
    log_file_path = args.logfile
    logging = configure_logger(log_file=log_file_path, console=False, log_level="INFO")

    processed_path = args.processeddatasetpath
    HOUSING_PATH = args.rawdatasetpath
    HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

    logging.info("Fetching housing data.")
    fetch_housing_data(HOUSING_URL, HOUSING_PATH)

    logging.info("Loading housing data.")
    housing = load_housing_data(HOUSING_PATH)

    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )
    logging.info("Splitting the dataset")
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    logging.info("Downloading the strat_train_set and strat_test_set")
    strat_train_set.to_csv(
        os.path.join(processed_path, "strat_train_set.csv"), index=False
    )
    strat_test_set.to_csv(
        os.path.join(processed_path, "strat_test_set.csv"), index=False
    )

    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

    logging.info("Downloading the train_set and test_set")
    train_set.to_csv(os.path.join(processed_path, "train_set.csv"), index=False)
    test_set.to_csv(os.path.join(processed_path, "test_set.csv"), index=False)

    compare_props = pd.DataFrame(
        {
            "Overall": income_cat_proportions(housing),
            "Stratified": income_cat_proportions(strat_test_set),
            "Random": income_cat_proportions(test_set),
        }
    ).sort_index()
    compare_props["Rand. %error"] = (
        100 * compare_props["Random"] / compare_props["Overall"] - 100
    )
    compare_props["Strat. %error"] = (
        100 * compare_props["Stratified"] / compare_props["Overall"] - 100
    )

    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    housing = strat_train_set.copy()
    housing.plot(kind="scatter", x="longitude", y="latitude")
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

    corr_matrix = housing.corr()
    corr_matrix["median_house_value"].sort_values(ascending=False)
    housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
    housing["population_per_household"] = housing["population"] / housing["households"]

    housing = strat_train_set.drop(
        "median_house_value", axis=1
    )  # drop labels for training set
    housing_labels = strat_train_set["median_house_value"].copy()
    logging.info("Downloading housing labels dataset")
    housing_labels.to_csv(
        os.path.join(processed_path, "housing_labels.csv"), index=False
    )

    imputer = SimpleImputer(strategy="median")

    housing_num = housing.drop("ocean_proximity", axis=1)

    imputer.fit(housing_num)
    X = imputer.transform(housing_num)

    housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing.index)
    housing_tr["rooms_per_household"] = (
        housing_tr["total_rooms"] / housing_tr["households"]
    )
    housing_tr["bedrooms_per_room"] = (
        housing_tr["total_bedrooms"] / housing_tr["total_rooms"]
    )
    housing_tr["population_per_household"] = (
        housing_tr["population"] / housing_tr["households"]
    )

    housing_cat = housing[["ocean_proximity"]]
    housing_prepared = housing_tr.join(pd.get_dummies(housing_cat, drop_first=True))
    logging.info("Downloading housing prepared dataset")
    housing_prepared.to_csv(
        os.path.join(processed_path, "housing_prepared.csv"), index=False
    )

    return True


if __name__ == "__main__":
    transform_data()
