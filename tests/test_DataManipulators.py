import os
import sys

import pytest
import pandas as pd
import numpy as np

from numpy.testing import assert_almost_equal

def add_file_paths(project_name: str):
    # figure out the path of the file we're runnning
    runpath = os.path.realpath(__file__)
    # find the root dir path
    rundir = runpath[:runpath.find(project_name) + len(project_name) + 1]
    # add the correct path to sys.path
    os.chdir(rundir + project_name)
    sys.path.append(rundir + project_name)

# Add the correct file path so that unit tests can find the correct non-package module/script to import.
add_file_paths(project_name='interview-test-final')
from util import DataManipulators


X = pd.DataFrame({'a': [0, 1.0, np.nan, np.nan],
                  'b': [2, 100.0, np.nan, np.nan],
                  'c': ['red', 'blue', 'green', np.nan]})
                   
y = pd.Series([0, 1, 0, 0])

a_list = [0, 0, 'a', 'b']


def test_error_checkDataFrame():
    # Test that correct errors are raised when checking dataframe type.
    with pytest.raises(TypeError):
        DataManipulators.checkDataFrameType(a_list)
        
    with pytest.raises(IndexError):
        DataManipulators.checkDataFrameType(pd.DataFrame())
    

def test_clean():
    # Test that clean function drops all NaN values.
    cleaned_df = DataManipulators.clean(X)
    
    assert_almost_equal(cleaned_df.isna().sum().sum(), 0)


def test_encode():
    # Test that all categorical data is encoded successfully.
    encoded_df = DataManipulators.encode(X)

    assert encoded_df.select_dtypes('object').empty