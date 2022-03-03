# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description

This project uses machine learning techniques to identify credit card customers that are most likely to churn. 
The project have the EDA and feature engineering required to train the models.
Results of the EDA are saved under 'images/eda'
The models inculded in the package under '/models' directory are .pkl files saved after training with the Random Forest Classifier and Logistic regression.
The completed project includes a Python package for a machine learning project that follows coding (PEP8) and engineering best practices for implementing software (modular, documented, and tested). The package can be run interactively or from the command-line interface (CLI).

## Dependencies

Before starting please make sure to install these libraries:

```
joblib
pandas
numpy
matplotlib
seaborn
scikit-learn
pylint

```
## Files in the Repo
### Folder structure

```
data
    bank_data.csv
images
    eda
        Churn_distribution.png
        Customer_age_distribution.png
        heatmap.png
        marital_status_distribution.png
        total_transaction_distribution.png
    results
        feature_importances.png
        logistic_results.png
        rf_results.png
        roc_curve_results.png
logs
    churn_library.log
models
    logistic_model.pkl
    rfc_model.pkl
churn_library.py
churn_notebook.ipynb
churn_script_logging_and_tests.py
README.md
```

### Running Files
To use the fucions import the churn_library module in your script.

`import churn_library as cls`

The project also includes unit tests for each function.
To run the tests run the following command:

` ipython churn_script_logging_and_test.py`