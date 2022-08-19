import os

from HousePricePrediction import train


def test_arguments_train():
    args = train.parse_arguments()
    assert os.path.exists(args.processed_dataset_path)
    assert os.path.exists(args.Pickle_path)
    assert os.path.exists("logs/train_log.txt")


def test_model_files():
    train.train_models()
    assert os.path.exists("artifacts/DecisionTree_reg_model.pkl")
    assert os.path.exists("artifacts/final_model.pkl")
    assert os.path.exists("artifacts/linear_reg_model.pkl")
