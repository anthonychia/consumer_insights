import logging
import json
import numpy as np
import pandas as pd
import os

from collections import namedtuple
from util import DataLoaders, DataManipulators, Predictors


# Set random seed to ensure reproducibility
np.random.seed(1234) 


def sort_file_paths(project_name: str):
    # figure out the path of the file we're runnning
    runpath = os.path.realpath(__file__)
    # trim off the bits we know about (i.e. from the root dir of this project)
    rundir = runpath[:runpath.find(project_name) + len(project_name) + 1]
    # change directory so we can use relative filepaths
    os.chdir(rundir + project_name)
    

def load_config():
    run_configuration_file = '../resources/consumer_insights.json'
    with open(run_configuration_file) as json_file:
        json_string = json_file.read()
        run_configuration = json.loads(json_string,
                                       object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))
    return run_configuration


if __name__ == '__main__':
    # Initialize logging
    logging.basicConfig(format="%(asctime)s;%(levelname)s;%(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
    logging.info('Starting classification program')

    # Actions: get into working directory, load project config, create dated directories
    sort_file_paths(project_name='consumer_insights')
    run_configuration = load_config()

    # Load the data by instantiating the FileDataLoader, handle file doesn't exist.
    filename = '../data/final_dataset.csv'
    data_loader = DataLoaders.FileDataLoader(filename)
    df = data_loader.load_data()
    df = df.drop('user_id', axis=1)  # dropping user_id, not an important feature
    
    # Clean the data and split into train and test.
    df = DataManipulators.encode(df)
    X_train, X_test, y_train, y_test = DataManipulators.split(df, target_label='great_customer_class')

    # Initiate a dictionary to gather all results.
    results = dict()

    # Define a list of models and respective hyperparameters to test.
    models = {'Baseline': Predictors.Classifier(algo='Baseline',
                                                strategy='stratified'),

              'RF': Predictors.Classifier(algo='RandomForest',
                                          min_impurity_decrease=0.000001,
                                          n_estimators=200),

              'RF_oversampling': Predictors.Classifier(algo='RandomForest',
                                                       min_impurity_decrease=0.000001,
                                                       n_estimators=200,
                                                       resampling_strategy='oversampling'),

              'SVM': Predictors.Classifier('SVM',
                                           C=100,
                                           gamma=0.025),

              'SVM_oversampling': Predictors.Classifier('SVM',
                                                        C=100,
                                                        gamma=0.025,
                                                        resampling_strategy='oversampling')}

    for key in models.keys():
        model = models[key]
        model.train(X_train, y_train)
        results[key] = model.evaluate(X_test, y_test)
    
    # Print results
    logging.info('Displaying results')
    print(pd.DataFrame.from_dict(results, orient='index'))
    
    logging.info('Completed program')
