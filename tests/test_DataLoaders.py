import os
import sys

import pytest
import pandas as pd
import numpy as np

from pandas.testing import assert_frame_equal
from pandas.errors import ParserError


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
from util import DataLoaders

test_df = pd.DataFrame({'a': [0, 1.0, np.nan, np.nan],
                        'b': [2, 100.0, np.nan, np.nan],
                        'c': ['red', 'blue', 'green', np.nan]})


def test_load_data():
    data_loader = DataLoaders.FileDataLoader('../tests/test_data.csv')
    df = data_loader.load_data()
    assert_frame_equal(df, test_df)


def test_error_fileNotFound():
    with pytest.raises(FileNotFoundError):
        data_loader = DataLoaders.FileDataLoader('../tests/dummy.csv')
        data_loader.load_data()


def test_error_parser():
    with pytest.raises(ParserError):
        data_loader = DataLoaders.FileDataLoader('../tests/broken.csv')
        data_loader.load_data()
