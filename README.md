# SpEmViz
#TODO - DOCUMENT VERY WELL THE INSTALLATION AND USAGE OF SPEMVIZ

# Installation
Clone the git repository by running in your preferred terminal emulator:
```sh
git clone https://github.com/delale/SpEmViz.git
```

Install [miniconda](https://docs.conda.io/projects/miniconda/en/latest/) or [Anaconda](https://www.anaconda.com/download).  
Open your preferred terminal emulator in the `SpEmViz` directory and install the spmviz conda environment:  
  
**MacOS**
```sh
conda env create -f spemviz_env_OSX.yml
```
  
**WinOS or Linux**
```sh
conda env create -f spemviz_env_Win_and_Linux.yml
```
  
Verify installation by running:
```sh
conda activate spemviz
which python3
```
It should return something along the lines of   
`/Users/your_user/miniconda3/envs/spemviz/bin/python3`  
or  
`/Users/your_user/anaconda3/envs/spemviz/bin/python3`

# Usage
To run the program open a terminal in the SpEmViz directory and run 
```sh
conda activate spemviz
```
To run the embedding projector on a dataset run:
```sh
python3 spemviz.py -f path_to_the_database -x feature variables -y metadata variables --log_dir path_to_logs_directory
```
You will see a CLI output similar to this:
```
TensorBoard Embedding Projector at: http://localhost:6006//#projector
Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
TensorBoard 2.13.0 at http://localhost:6006/ (Press CTRL+C to quit)
```
To open directly TensorBoard embedding projector Cmd+Click (MacOS) or CTRL+Click on the first link (`http://localhost:6006//#projector`) or copy and paste that link to your browser.  
If you open the second link, you can navigate to the embedding projector by clicking on the drop-down menu in the top-right corner of the page which says **INACTIVE**, scroll down and click on **PROJECTOR**.
![Alt text](<screenshots/Screenshot 2023-09-12 at 16.44.30.png>)
![Alt text](<screenshots/Screenshot 2023-09-12 at 16.44.51.png>)
When you want to exit you can simply close the embedding projector page and in your terminal emulator press CTRL+C.
# Arguments
- `--filepath` or `-f`: path to the data.
- `--features` or `-x`: embedding features that can have several forms (default: `None`):
  - `-x x1 x2 x3`: will select the features by column names ('x1', 'x2', 'x3')
  - `-x 1 3 4`: will select the features by column indices (1, 3, 4); *remember that python indices start at 0, so the first column will have index 0*
  - `-x "(0, 4)"`: will select all columns by indices within the range [0, 4] (0, 1, 2, 3,4)
  - `-x all`: will take all columns of the data
  - `-x None`: will take all columns of the data which have not been specified in `--metavars`
- `--metavars` or `-y`: metadata variables that can have several forms (default: `None`):
  - `-y y1 y2 y3`: will select the metadata by column names ('y1', 'y2', 'y3')
  - `-y 1 3 4`: will select the metadata by column indices (1, 3, 4); *remember that python indices start at 0, so the first column will have index 0*
  - `-y "(0, 4)"`: will select all columns by indices within the range [0, 4] (0, 1, 2, 3,4)
  - `-y infer`: will take all columns of the data which have not been specified in `--features`
  - `-y None`: no metadata
- `--log_dir`: (optional) path to the logs directory (default: `./logs/`). The logs directory is where the data for TensorBoard is set-up.

# Example Usage
Given a database with 10 columns of which the last 6 are features (columns 5 through 9) and the first four are metadata variables: filename, speaker, sex, utterance.  
The database is located in my SpEmViz working directory at `data/db.csv`
To run the embedding projector with just the features by running:
```sh
python3 spemviz.py -f ./data/db.csv -x "(5, 9)" -y None
```
To run the embedding projector with the features and only `sex` and `speaker` by running:
```sh
python3 spemviz.py -f ./data/db.csv -x "(5, 9)" -y sex speaker
```
or
```sh
python3 spemviz.py -f ./data/db.csv -x "(5, 9)" -y 2 1
```
To run the embedding projector with the features and all of the metadata variables by running:
```sh
python3 spemviz.py -f ./data/db.csv -x "(5, 9)" -y infer
```
To run the embedding projector with only `F0`, `F1`, and `HNR` as features and all of the metadata variables by running:
```sh
python3 spemviz.py -f ./data/db.csv -x F0 F1 HNR -y "(0 3)"
```