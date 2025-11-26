from setuptools import setup, find_packages

setup(
    name="mlproject",
    packages=find_packages(),
    install_requires=["numpy", 
                      "pandas", 
                      "scikit-learn", 
                      "seaborn", 
                      "matplotlib",
                      "catboost",
                      "xgboost",
                      "flask"
                      ]
)