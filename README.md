# voxplorer
![Embedding Projector example](screenshots/embedding_dark.png)
| Table of Contents               |
| ------------------------------- |
| [Installation](#installation)   |
| [Usage](#usage)                 |
| [Arguments](#arguments)         |
| [Example Usage](#example-usage) |
| [Tips & Tricks](#tips--tricks)  |
| [Caveats](#caveats)             |
## Installation
Clone the git repository by running in your preferred terminal emulator:
```sh
git clone https://github.com/delale/voxplorer.git
```

Install [miniconda](https://docs.conda.io/projects/miniconda/en/latest/) or [Anaconda](https://www.anaconda.com/download).  
Open your preferred terminal emulator in the `voxplorer` directory and install the `voxplorer` conda environment:  

**Linux**
```sh
conda env create -f voxplorer_env_Linux.yml
```
:exclamation: This installation has been tested on x86 64-bit Ubuntu only.

**WinOS**
```sh
conda env create -f voxplorer_env_WinOS.yml
```
:exclamation: This installation has not been tested thoroughly yet.

**OSX**
```sh
conda env create -f voxplorer_env_OSX.yml
```
Sometimes an error during the installation of tensorflow and tensorflow-metal may occur in particular with older Intel x86 Macs:
```sh
Pip failed

ERROR: Could not find a version that satisfices the requirement tensorflow-metal (from versions: None)
ERROR: No matching distribution found for tensorflow-metal
```
In this case run the following commands:
```sh
conda activate voxplorer
SYSTEM_VERSION_COMPAT=0 pip install tensorflow tensorflow-metal
```
  
**Verify installation by running:**
```sh
conda activate voxplorer
which python3
```
It should return something along the lines of  
`/Users/your_user/miniconda3/envs/voxplorer/bin/python3`  
or  
`/Users/your_user/anaconda3/envs/voxplorer/bin/python3`

## Main Usage
To run the program open a terminal emulator in the voxplorer directory and run 
```sh
conda activate voxplorer
```
To open the voxplorer GUI run:
```sh
python3 voxplorer.py
```
A GUI window should open:  
![Main GUI window](<screenshots/mainGUI.png>)  
Select your preferred mode... and have fun exploring!

### Visualizer Mode:
With this mode you can visualize in the embedding projector a table of previously extracted features. When selected you will be prompted to select a table; after the table has been selected, the box below will show you a list containing all of the table columns. You can click on each variable that you would like to use as metadata variables. By checking the `Add 'selection' column` checkbox voxplorer will include a `selection` column in the metadata, which by default contains only 0s and can be used to filter the data from within the visualizer. 
> Metadata variables are usually categorical variables used for colouring and labelling in the embedding visualization. Please select all variables that are not to be used as features in the calculation of the reduced dimension space.  
 
![Visualizer mode window](<screenshots/visGUI.png>)

### Feature Extraction and Visualization Mode:
With this mode you can extract acoustic features from a set of audio files directly within voxplorer. To do this, all you need to do is select the directory where the audio files (as of now only .WAV files are compatible) are contained.
> The directory can contain also other files (e.g. Praat .TextGrid); voxplorer will simply ignore files that are not audio files.  

> For ease of use, we decided to include the possibility of extracting features from one audio file only (using the `Browse file` button for example). It does not make a lot of sense to project 1 observation, and the projector is unable to visualize only 1 observation at a time, but needs a group of observations to work properly.  

After having selected the input files, you can specify an output directory and filename if you would like to save the extracted features to a table.  
You can then select weather you would like to extract `Speaker embeddings` (ECAPA-TDNN VoxCeleb2 X-vectors) or a set of other acoustic features (`Feature extraction`) by selecting the corresponding radio button.  
![Feature extraction and visualization mode window](<screenshots/featextractGUI.png>)

#### Specifying metadata:


### Project!
When finished with the setup (in any window) you can click `Continue` and a `Projector` window will open. By clicking on `Project!` a new browser tab will open automatically on the embedding projector local webpage.  

To exit you can simply close the browser tab.
> A `Stop projector` button will appear in the `Projector` window, but unfortunately this is as of now not functional.

In the embedding projector you can then use the several different tools (rectangle selection, selecting by label for e.g. sex, clicking on point and selecting n nearest neighbours) and then on the left tab, navigate to `Edit by` and change the metadata to edit to `selection`. Then change the value to your preferred selection value (e.g. 1). You can then download the edited metadata by clicking on download. You can then select only those rows with the edited `selection` value and use the key variable (e.g. `filename`) to filter the original data. This works well only when including a key ID variable that is unique for each row in the original dataframe. Theoretically, you can also use the indices of the selected rows to filter the dataframe as they should remain the same (*caveat: if the original contains NAs, the indices will be relative only to the NA-filtered data*).

## Caveats
- Data that contains NAs: a Warning will be raised and the rows containing NA values will be removed (otherwise TensorBoard will raise an error where it is unable to transorm str to float64).