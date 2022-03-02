'''

This file includes the tests
for churn_library.py modules, results
of the test logs are found in ./logs/churn_library.log

Author: Tesnim Hadhri
Date: 02/03/2022

'''

import os
import logging
from shutil import ExecError
import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''

    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err

    return df


def test_eda(perform_eda, df):
    '''
    test perform eda function
    '''

    try:
        perform_eda(df)
        logging.info("Testing perform_eda: SUCCESS")
    except ExecError as err:
        logging.error(
            "Testing perform_eda: function failed to execute.")
        raise err

    try:
        assert os.path.isfile("./images/eda/churn_distribution.png")
        assert os.path.isfile("./images/eda/customer_age_distribution.png")
        assert os.path.isfile("./images/eda/marital_status_distribution.png")
        assert os.path.isfile(
            "./images/eda/total_transaction_distribution.png")
        assert os.path.isfile("./images/eda/heatmap.png")
    except AssertionError as err:
        logging.info('Test perfom_eda failed: Not all figures are saved')

        raise err


def test_encoder_helper(encoder_helper, df):
    '''
    test encoder helper
    '''

    category_lst = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    try:
        encoder_helper(df, category_lst)
        logging.info("Testing encoder_helper: SUCCESS")
    except ExecError as err:
        logging.error("Testing encoder_helper: Failed")
        raise err

    try:
        assert df.shape[1] == 28
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: Failed: The columns shape of the data frame is unvalid")
        raise err
    return df


def test_perform_feature_engineering(perform_feature_engineering, df):
    '''
    test perform_feature_engineering
    '''
    try:
        X_train, X_test, y_train, y_test = perform_feature_engineering(df)
        logging.info("Testing perform_feature_engineering: SUCCESS")

    except ExecError as err:
        logging.error(
            "Testing perform_feature_engineering: Failed")
        raise err

    try:
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert y_train.shape[0] > 0
        assert y_test.shape[0] > 0
        assert X_train.shape[0] == y_train.shape[0]
        assert X_test.shape[0] == y_test.shape[0]
    except ExecError as err:
        logging.error(
            "Testing perform_feature_engineering: Failed: Data split shapes are unvalid")
        raise err

    return X_train, X_test, y_train, y_test


def test_train_models(train_models, X_train, X_test, y_train, y_test):
    '''
    test train_models
    '''

    try:
        train_models(X_train, X_test, y_train, y_test)
        logging.info("test_train_models: SUCCESS")
    except ExecError as err:
        logging.error("Testing Train_models: Failed.")
        raise err

    try:
        assert os.path.isfile("./images/eda/churn_distribution.png")
        assert os.path.isfile("./images/eda/customer_age_distribution.png")

        logging.info('Test Train_models: SUCCESS')
    except AssertionError as err:
        logging.info('Test perfom_eda failed: models were not saved.')

        raise err


if __name__ == "__main__":
    df = test_import(cls.import_data)
    test_eda(cls.perform_eda, df)
    test_encoder_helper(cls.encoder_helper, df)
    X_train, X_test, y_train, y_test = test_perform_feature_engineering(
        cls.perform_feature_engineering, df)
    test_train_models(cls.train_models, X_train, X_test, y_train, y_test)
