# Getting started

We highly recommend installing the required packages in a Python virtual environment, e.g. using conda OR venv, the minimum requirement for Python is `3.10`. System specific instructions are below.

The exercise is not plain Python, but running in a Jupyter notebook. It can be executed either through the VS Code IDE (recommended) or Jupyter lab. In any case, before running, you'll need to select a kernel (Python plus virtual env). Chose it according to the virtual environment you define/use below.

To run the notebook using Jupyter (installed with below install as well), first launch Jupyter notebook with ```jupyter notebook``` from the command line. This launches a GUI in your browser, where you can navigate to the notebook (the .ipynb file) and start running the code.

# Setting up the virtual environment
## For Linux/ macOS users

In your terminal/shell, navigate to the folder of this exercise. Then:

### venv:

(If missing on Linux: ```sudo apt install python3-venv```)

```
python3 -m venv amr
source amr/bin/activate
pip install -r requirements.txt
```

### conda:

Follow https://www.anaconda.com/docs/getting-started/miniconda/install#quickstart-install-instructions

```
conda create -n amr python=3.10
conda activate amr
pip install -r requirements.txt
```

## For Windows users:

Install Python (>=3.10) if you haven't already, see e.g. https://learn.microsoft.com/en-us/windows/dev-environment/python?tabs=winget for instructions and https://www.python.org/downloads/windows/ for downloads.
In your terminal/shell, navigate to the folder of this exercise. Then:

### venv:
```
python -m venv amr (or py -m venv amr)
amr\Scripts\activate
pip install -r requirements.txt
```

### conda:

Follow https://www.anaconda.com/docs/getting-started/miniconda/install#quickstart-install-instructions

```
conda create -n amr python=3.10
conda activate amr
conda install pinocchio numpy example-robot-data example-robot-data-loaders meshcat-python matplotlib jupyterlab -c conda-forge
```