import logging
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

from sklearn.model_selection import train_test_split


def checkDataFrameType(data):
    # Check data is in the right format.
    if not isinstance(data, pd.DataFrame):
        logging.info('Data is not in pandas dataframe format.')
        raise TypeError('Data is not in pandas dataframe format.')
    elif data.empty:
        logging.info('Dataframe is empty.')
        raise IndexError('Dataframe is empty.')
    else:
        pass


def clean(data):
    # Drop rows with missing values.
    checkDataFrameType(data)
    
    logging.info('Cleaning data.')
    return data.dropna()

    
def split(data, target_label, train_test_ratio=0.75):
    # Split data into train and test sets
    checkDataFrameType(data)
    
    logging.info('Splitting data into train and test.')
    # Split into features and labels. Assuming last row are the labels.
    X, y = data.drop(target_label, axis=1), data[target_label].astype(int)
    
    # Split data using defined train_test_ratio.
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        train_size=train_test_ratio,
                                                        test_size=1-train_test_ratio)
    
    return X_train, X_test, y_train, y_test


def encode(data):
    # One-hot encode the data to change categorical data into binary variables.
    checkDataFrameType(data)
    
    logging.info('Encoding inputs.')
    return pd.get_dummies(data, dummy_na=True)
