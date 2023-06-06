# Artifacts using Vertex ML Pipelines
This repo contains code for an example ML Pipeline (Kubeflow) on Vertex AI. Instead of using the default Artifact URI as provided by Vertex AI, the file location is set manually. 

## Setting up environment

To set up the environment, create a new venv and install the requirements.

``` bash
python3 -m venv pipeline-env
source pipeline-env/bin/activate
pip install -r requirements.txt
``` 

If you later you encounter issues with protobuf, try uninstalling and installing it manually:

``` bash
pip uninstall protobuf
pip install protobuf
``` 

## Running pipelines
There are two pipelines; one where data preprocessing is done in the [pipeline](pipeline.py) and one where a preprocessed dataset is loaded from an existing Artifact: [train_only_pipeline](train_only_pipeline.py).

To run the pipeline, first you need to specify some variables, in pipeline file you're running.
1. project_id  
   Google Cloud Project name the pipeline will run on
2. pipeline_root_path  
    Root location to store pipeline files to; must be a path to a folder on Google Cloud Storage
3. data_file_location (only for [train_only_pipeline](train_only_pipeline.py))  
    URI of preprocessed Dataset Artifact; must be valid file on Google Cloud Storage

To run the pipeline run one of these.
``` bash
python pipeline.py
python train_only_pipeline.py
```