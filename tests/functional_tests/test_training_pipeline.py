import os

from HousePricePrediction import train


def test_train():
    train.train_models()
    assert os.path.exists("artifacts/final_model.pkl")
