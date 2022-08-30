import os

from HousePricePrediction import score


def test_arguments():
    args = score.parse_arguments()
    assert os.path.exists(args.model_folder)
    assert os.path.exists(args.processed_path)
    assert os.path.exists(args.log_file)
