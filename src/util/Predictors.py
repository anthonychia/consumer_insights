import logging
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
# from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score
from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


class Model(ABC):

    def __init__(self):
        super().__init__()
        logging.info('Initializing model')

    @abstractmethod
    def train(self):
        # Train a standard model (without hyperparameter search)
        logging.info('Training model')

    @abstractmethod
    def predict(self):
        # Predict using trained model
        logging.info('Doing predictions')

    @abstractmethod
    def evaluate(self):
        # Predict using trained model
        logging.info('Evaluating model ')
       

class Classifier(Model):

    def __init__(self, algo='Baseline', resampling_strategy=None, **kwargs):
        # Initiate appropriate model.
        super().__init__()
        self.algo = algo

        # Define modelling steps.
        steps = [('imputer', SimpleImputer())]  # Use simpler mean imputation.

        # Add more modelling steps according to selected algorithm.
        if resampling_strategy == 'oversampling':
            steps.append(('sampler', RandomOverSampler()))
        elif resampling_strategy == 'undersampling':
            steps.append(('sampler', RandomUnderSampler()))
        
        if algo == 'RandomForest':
            steps.append(('model', RandomForestClassifier(**kwargs)))
            self.model = Pipeline(steps)
        elif algo == 'SVM':
            steps.append(('scaler', MinMaxScaler()))
            steps.append(('model', SVC(**kwargs)))
            self.model = Pipeline(steps)
        elif algo == 'Baseline':
            self.model = DummyClassifier(**kwargs)
        else:
            raise ValueError(f"Unknown algorithm type: {self.algo}, expected one of ('RandomForest', 'SVM', 'Baseline').")

    def train(self, X_train, y_train):
        # Train the model
        logging.info(f'Training model: {self.algo}')

        self.model.fit(X_train, y_train.values.reshape(len(y_train)))

    def predict(self, X):
        # Make predictions
        logging.info('Doing predictions')
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        # Compute model evaluation results
        logging.info(f'Evaluating model: {self.algo}')
        y_pred = self.model.predict(X_test)
        
        results = {  # 'cm': confusion_matrix(y_test, y_pred).tolist(),
                   'precision': precision_score(y_test, y_pred),
                   'recall': recall_score(y_test, y_pred),
                   'f1': f1_score(y_test, y_pred),
                   'accuracy': accuracy_score(y_test, y_pred)}
        
        # Obtain feature importances for Random Forest model
        if self.algo == 'RandomForest':
            importances = sorted(list(zip(X_test.columns, self.model.steps[-1][1].feature_importances_)),
                                 key=lambda x: x[1],
                                 reverse=True)
            results['importances'] = importances
            
        return results
