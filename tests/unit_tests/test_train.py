import os

from HousePricePrediction import train


def test_arguments_train():
    args = train.parse_arguments()
    assert os.path.exists(args.processed_dataset_path)
    assert os.path.exists(args.Pickle_path)
    assert os.path.exists(args.logfile)


def test_model_files():
    train.main("Linear Regressor")
    assert os.path.exists("artifacts/linear_reg_model.pkl")
    train.main("Decision Tree Regressor")
    assert os.path.exists("artifacts/DecisionTree_reg_model.pkl")
    train.main("RandomSearch RandomForest Regressor")
    assert os.path.exists("artifacts/RandomSearch_RandomForest_Model.pkl")
    train.main("GridSearchCV RandomForest Regressor")
    assert os.path.exists("artifacts/final_model.pkl")
