## Setup

Make sure the Anaconda distribution is installed on your machine.

1. Create and activate a new conda environment with Python 3.10
```shell
$ conda create -n <environment_name> python=3.10
$ conda activate <environment_name>
```
2. Install dependencies
* recommenders - `pip install recommenders`
* mlflow - `conda install mlflow`; make sure the installed version is at least 2.16 to avoid problems with Databricks integration that are present in older versions
* tensorflow - `pip install tensorflow==2.12.0`
3. Run Jupyter Notebook
```shell
jupyter notebook
```
4. Deactivate the conda environment when finished:
```shell
$ conda deactivate
```

To completely remove the conda environment execute:
```shell
$ conda remove -n <environment_name> --all
```

## Development

Once the conda <environment_name> has been successfully [set up](#setup), the only requirement is to activate
that environment:
```shell
$ conda activate <environment_name>
```
and (usually) starting the Jupyter Notebook:
```shell
jupyter notebook
```
