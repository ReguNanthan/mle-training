import os
import tarfile

import numpy as np
from six.moves import urllib

import re
import argparse

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.impute import SimpleImputer

from logger import configure_logger

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

import mlflow

import warnings

warnings.filterwarnings("ignore")


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


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):  # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X):
        rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[
                X, rooms_per_household, population_per_household, bedrooms_per_room
            ]

        else:
            return np.c_[X, rooms_per_household, population_per_household]

    def columns(self):
        if self.add_bedrooms_per_room:
            cols = [
                "rooms_per_household",
                "population_per_household",
                "bedrooms_per_room",
            ]
        else:
            cols = ["rooms_per_household", "population_per_household"]
        return cols


def get_feature_names_from_column_transformer(col_trans):
    """Get feature names from a sklearn column transformer.

    The `ColumnTransformer` class in `scikit-learn` supports taking in a
    `pd.DataFrame` object and specifying `Transformer` operations on columns.
    The output of the `ColumnTransformer` is a numpy array that can used and
    does not contain the column names from the original dataframe. The class
    provides a `get_feature_names` method for this purpose that returns the
    column names corr. to the output array. Unfortunately, not all
    `scikit-learn` classes provide this method (e.g. `Pipeline`) and still
    being actively worked upon.

        NOTE: This utility function is a temporary solution until the proper fix is
    available in the `scikit-learn` library.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder as skohe

    # SimpleImputer has `add_indicator` attribute that distinguishes it from other transformers
    # Encoder had `get_feature_names` attribute that distinguishes it from other transformers
    # The last transformer is ColumnTransformer's 'remainder'
    col_name = []
    for transformer_in_columns in col_trans.transformers_:
        is_pipeline = 0
        raw_col_name = list(transformer_in_columns[2])

        if isinstance(transformer_in_columns[1], Pipeline):
            # if pipeline, get the last transformer
            transformer = transformer_in_columns[1].steps[-1][1]
            is_pipeline = 1
        else:
            transformer = transformer_in_columns[1]

        try:
            if isinstance(transformer, str):
                if transformer == "passthrough":
                    names = transformer._feature_names_in[raw_col_name].tolist()

                elif transformer == "drop":
                    names = []

                else:
                    raise RuntimeError(
                        f"Unexpected transformer action for unaccounted cols :"
                        f"{transformer} : {raw_col_name}"
                    )

            elif isinstance(transformer, skohe):
                names = list(transformer.get_feature_names(raw_col_name))

            elif isinstance(transformer, SimpleImputer) and transformer.add_indicator:
                missing_indicator_indices = transformer.indicator_.features_
                missing_indicators = [
                    raw_col_name[idx] + "_missing_flag"
                    for idx in missing_indicator_indices
                ]

                names = raw_col_name + missing_indicators

            else:
                names = list(transformer.get_feature_names())

        except AttributeError as error:
            names = raw_col_name
        if is_pipeline:
            names = [f"{transformer_in_columns[0]}_{col_}" for col_ in names]
        col_name.extend(names)

    return col_name


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

    housing = strat_train_set.drop(
        "median_house_value", axis=1
    )  # drop labels for training set
    housing_labels = strat_train_set["median_house_value"].copy()
    logging.info("Downloading housing labels dataset")
    housing_labels.to_csv(
        os.path.join(processed_path, "housing_labels.csv"), index=False
    )
    mlflow.log_artifact(os.path.join(processed_path, "housing_labels.csv"))

    housing_num = housing.drop("ocean_proximity", axis=1)

    attr_adder = CombinedAttributesAdder()
    cols = attr_adder.columns()

    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("attribs_adder", CombinedAttributesAdder()),
            ("std_scaler", StandardScaler()),
        ]
    )

    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]

    full_pipeline = ColumnTransformer(
        [
            ("num", num_pipeline, num_attribs),
            ("cat", OneHotEncoder(), cat_attribs),
        ]
    )

    housing_prepared_numpyarray = full_pipeline.fit_transform(housing)

    column_names = get_feature_names_from_column_transformer(full_pipeline)

    house_prep = (
        pd.DataFrame(housing_prepared_numpyarray[:, :8], columns=column_names[:8])
    ).join(
        (pd.DataFrame(housing_prepared_numpyarray[:, 8:11], columns=cols)).join(
            pd.DataFrame(housing_prepared_numpyarray[:, 11:], columns=column_names[8:])
        )
    )

    for i in range(len(house_prep.columns)):
        if "num" in house_prep.columns[i]:
            house_prep.rename(
                columns={
                    house_prep.columns[i]: re.sub("num_", "", house_prep.columns[i])
                },
                inplace=True,
            )

    housing_prepared = house_prep

    logging.info("Downloading housing prepared dataset")
    housing_prepared.to_csv(
        os.path.join(processed_path, "housing_prepared.csv"), index=False
    )

    mlflow.log_artifact(os.path.join(processed_path, "housing_prepared.csv"))

    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()

    data = full_pipeline.fit_transform(X_test)

    l1 = get_feature_names_from_column_transformer(full_pipeline)
    X_prep = (
        pd.DataFrame(data[:, :8], columns=l1[:8])
        .join(pd.DataFrame(data[:, 8:11], columns=cols))
        .join(pd.DataFrame(data[:, 11:], columns=l1[8:]))
    )

    for i in range(len(X_prep.columns)):
        if "num" in X_prep.columns[i]:
            X_prep.rename(
                columns={X_prep.columns[i]: re.sub("num_", "", X_prep.columns[i])},
                inplace=True,
            )

    xtest_prepared = X_prep  # .join(y_test)
    logging.debug("Performed all the preprocessing on test data.")
    logging.debug("Downloading the test data")

    xtest_prepared.to_csv(
        os.path.join(processed_path, "xtest_prepared.csv"), index=False
    )

    mlflow.log_artifact(os.path.join(processed_path, "xtest_prepared.csv"))

    y_test.to_csv(os.path.join(processed_path, "y_test_prepared.csv"), index=False)

    mlflow.log_artifact(os.path.join(processed_path, "y_test_prepared.csv"))

    return True


def main():
    transform_data()


if __name__ == "__main__":
    main()
