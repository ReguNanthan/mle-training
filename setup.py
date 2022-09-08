from setuptools import setup  # , find_packages

# read the contents of your README file
from pathlib import Path

this_directory = Path(__file__).parent

long_description = (this_directory / "README.md").read_text()


setup(
    name="HousePricePrediction",
    version="0.5.1",
    author="Regu Nanthan K",
    author_email="regunanthan.k@tigeranalytics.com",
    # package_dir={'': 'src/HousePricePrediction'},
    packages=["HousePricePrediction"],
    package_dir={"HousePricePrediction": "src/HousePricePrediction"},
    long_description=long_description,
    long_description_content_type="text/markdown"
    # packages= find_packages(where="src")
    # scripts=['bin/script1','bin/script2']
)
