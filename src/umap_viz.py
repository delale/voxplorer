"""UMAP visualisation of MFCCs data.
"""

import os

import numpy as np
import pandas as pd
import umap
import seaborn as sns

def main(data_path:str, target_var, n_dim=2, target_weight=0.6):
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"{data_path} does not exist.")
    
    # load data
    df = pd.read_csv(data_path)

    # def targets and features
    if type(target_var) is int:
        target_var = df.columns[target_var]
    if type(target_var) is str:
        df.loc[target_var] = df.loc[target_var].astype('category')
        y = df.loc[target_var].cat.codes.values
        labels = df.loc[target_var].to_numpy()
        X = df.drop(target_var, axis=1).to_numpy()
    else:
        raise ValueError(f"target_var has to be of either type int or str.")
    

    # init reducer
    reducer = umap.UMAP(
        n_components=n_dim, n_neighbors=15, metric='manhattan',
        target_weight=target_weight
        )

    # fit and transform
    redX = reducer.fit_transform(X, y)

    # prepare for plot
    reduced_df = {f'dim{i}': redX[:, i] for i in range(redX.shape[1])}
    reduced_df[target_var] = labels
    reduced_df = pd.DataFrame(reduced_df)

    # PLOTTING