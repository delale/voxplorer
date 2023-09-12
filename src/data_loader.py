"""Data loading module for embedding projector.
Data is loaded using a path and then divided into the variables and features.

----------
Alessandro De Luca - 09/2023
"""
import numpy as np
import pandas as pd


def load_data(features: list, metavars: list, path_to_data: str):
    """Splits the data into features and metadata.

    Args:
        features: Features variable names or indices list. 
            If None, all are taken except those listed in metavars.
            If 'all', all are taken. 
        metavars: Metadata variable names or indices list. 
            If None, metadata variables are excluded. 
            If 'infer', metadata variables are inferred as the ones not selected as features.
        path_to_data: Path to the data table.

    Returns:
        X [np.ndarray]: Features array.
        metadata [np.ndarray]: Metadata values (if metavars is not None).
        metavars [list]: Metadata variable names (if metavars is not None)

    Caveats:
        If both metavars and features are None, all of the data is taken as features.
        If features is None and
    """
    # Find the appropriate separator character
    with open(path_to_data, 'r') as f:
        line = f.readline()
        if len(line.split(',')) > 1:
            sep = ','
        elif len(line.split('\t')) > 1:
            sep = '\t'
        elif len(line.split(' ')) > 1:
            sep = ' '
        elif len(line.split(';')) > 1:
            sep = ';'
        else:
            sep = input("What is the separator character for the data?")

    df = pd.read_csv(path_to_data, sep=sep)

    # Feature selection and metadata variable selection
    # Do all checks on features and metavars to understand what needs to go where
    if (features is None and metavars is None) or features == 'all':
        X = df.to_numpy()
        return X, None, None

    if features is None and metavars is not None:
        if type(metavars) is str:
            if metavars == 'infer':
                raise ValueError("Cannot infer metavars when features is None")
        elif type(metavars[0]) is int:
            metavars = df.columns[metavars].to_list()

        X = df.loc[:, ~df.columns.isin(metavars)].astype(np.float64).to_numpy()
        Y = df.loc[:, metavars].astype(str).to_numpy()
        return X, Y, metavars

    if features is not None and metavars is None:
        if type(features[0]) is int:
            features = df.columns[features]

        X = df.loc[:, features].astype(np.float64).to_numpy()
        return X, None, None

    if features is not None and metavars is not None:
        if metavars == 'infer':
            if type(features[0]) is int:
                features = df.columns[features]

            X = df.loc[:, features].astype(np.float64).to_numpy()
            Y = df.loc[:, ~df.columns.isin(features)]
            metavars = Y.columns.to_list()
            Y = Y.astype(str).to_numpy()
            return X, Y, metavars
        else:
            if type(features[0]) is int:
                features = df.columns[features]
            if type(metavars[0]) is int:
                metavars = df.columns[metavars].to_list()

            X = df.loc[:, features].astype(np.float64).to_numpy()
            Y = df.loc[:, metavars].astype(str).to_numpy()
            return X, Y, metavars


if __name__ == '__main__':
    # Testing data_load
    PTD = 'testing/data_load_test.csv'  # running from main dir

    print(f"{'-'*10}\nTEST1: features=[0,1,2,3,4], metavars=['z']")
    x, y, mv = load_data(
        features=list(range(0, 5)), metavars=['z'], path_to_data=PTD
    )
    print(f"X.shape: {x.shape}; expected: (5,5)")
    print(f"Y.shape: {y.shape}; expected: (5,) or (5,1)")
    print(f"Metavars: {mv}; expected: ['z']\n\n")

    print(f"{'-'*10}\nTEST2: features=[0,1,2,3,4], metavars='infer'")
    x, y, mv = load_data(
        features=list(range(0, 5)), metavars='infer', path_to_data=PTD
    )
    print(f"X.shape: {x.shape}; expected: (5,5)")
    print(f"Y.shape: {y.shape}; expected: (5,3)")
    print(f"Metavars: {mv}; expected: ['y0', 'y1' 'z']\n\n")

    print(f"{'-'*10}\nTEST3: features=[0,1,2,3,4], metavars=None")
    x, y, mv = load_data(
        features=list(range(0, 5)), metavars=None, path_to_data=PTD
    )
    print(f"X.shape: {x.shape}; expected: (5,5)")
    print(f"Y is {type(y)}; expected: None")
    print(f"Metavars is {type(mv)}; expected: None\n\n")

    PTD = 'testing/data_load_test.scsv'  # testing also the infer separator part

    print(f"{'-'*10}\nTEST4: features=None, metavars=['z']")
    x, y, mv = load_data(
        features=None, metavars=['z'], path_to_data=PTD
    )
    print(f"X.shape: {x.shape}; expected: (5,7)")
    print(f"Y.shape: {y.shape}; expected: (5,) or (5,1)")
    print(f"Metavars: {mv}; expected: ['z']\n\n")

    print(f"{'-'*10}\nTEST5: features='all', metavars=['z']")
    x, y, mv = load_data(
        features='all', metavars=['z'], path_to_data=PTD
    )
    print(f"X.shape: {x.shape}; expected: (5,8)")
    print(f"Y is {type(y)}; expected: None")
    print(f"Metavars is {type(mv)}; expected: None\n\n")
