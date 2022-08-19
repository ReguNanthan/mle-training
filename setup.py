from setuptools import setup, find_packages

setup(
    name="HousePricePrediction",
    version="0.2.1",
    author="Regu Nanthan K",
    author_email="regunanthan.k@tigeranalytics.com",
    # package_dir={'': 'src/HousePricePrediction'},
    packages=["HousePricePrediction"],
    package_dir={"HousePricePrediction": "src/HousePricePrediction"}
    # packages= find_packages(where="src")
    # scripts=['bin/script1','bin/script2']
)
