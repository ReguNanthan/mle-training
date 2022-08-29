import os

from HousePricePrediction import train


def test_train():
    train.main()
    assert os.path.exists("artifacts/final_model.pkl")
