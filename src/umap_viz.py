"""UMAP visualisation of MFCCs data.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
import seaborn as sns

def main(data_path:str, target_var, n_dim=2, target_weight=0.6):
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"{data_path} does not exist.")
    
    n_plot_dim=n_dim
    if n_dim == 3:
        raise Warning("3D plotting not coming soon. Defaulting to 2D.")
        n_plot_dim = 2
    elif n_dim > 3:
        raise Warning(f"Cannot plot in more than 2D. UMAP embedding will be created using {n_dim} dimensions, but only the first 2 dimensions will be plotted.")
        n_plot_dim = 2

    # load data
    df = pd.read_csv(data_path)

    # def targets and features
    if type(target_var) is int:
        target_var = df.columns[target_var]
    if type(target_var) is str:
        df[target_var] = df[target_var].astype('category')
        y = df[target_var].cat.codes.values
        labels = df[target_var].to_numpy()
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
    reduced_df = {f'dim{i+1}': redX[:, i] for i in range(n_plot_dim)}
    reduced_df[target_var] = labels
    reduced_df = pd.DataFrame(reduced_df)

    # PLOTTING
    ## pairs plot for dimensions could be interenting with more than 3D.
    if n_plot_dim == 2:
        fig, ax = plt.subplots(1,1, figsize=(15,15))
        sns.scatterplot(
            data=reduced_df, x='dim1', y='dim2', hue=target_var, ax=ax
        )
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'data', type=str, help='Path to the data table (.csv containing 1 target variable and features.)'
    )
    parser.add_argument(
        'target_var', help='Name or index of target variable column (index starst from 0).'
    )
    parser.add_argument(
        'dim', type=int, help='Number of dimensions to reduce using UMAP (default=2).', default=2
    )
    parser.add_argument(
        'target_weight', type=float, help='Target weight for UMAP (default=0.6).', default=0.6
    )
    args = parser.parse_args()

    try:
        args.target_var = int(args.target_var)
    except ValueError:
        pass
    
    main(data_path=args.data, target_var=args.target_var, n_dim=args.dim, target_weight=args.target_weight)