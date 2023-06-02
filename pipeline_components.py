from kfp.v2.dsl import component, Dataset, Metrics, Model, Input, Output


@component(
    # This component performs preprocessing on the default sklearn
    # iris dataset and to creates an Artifact of the preprocessed
    # dataset.
    packages_to_install=[
        "scikit-learn==1.2.1",
        "pandas==1.5.3",
        "google-cloud-storage==1.44.0",
        "gcsfs==2023.1.0",
        "fsspec==2023.1.0",
    ],
    base_image="python:3.10",
    output_component_file="./compiled_pipelines/preprocess_data.yaml",
)
def preprocess_data(
    preprocessed_dataset: Output[Dataset], pipeline_root: str, run_name: str
) -> None:
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.datasets import load_iris

    # Load iris dataset and assign to pandas dataframe
    data = load_iris()
    df = pd.DataFrame(data=data.data, columns=data.feature_names)
    df.columns = [x.replace(" (cm)", "").replace(" ", "_") for x in df.columns]

    # Scale every column
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_df, columns=df.columns)

    # Add target column
    scaled_df["target"] = data.target
    scaled_df["target"] = scaled_df.target.apply(lambda x: data.target_names[x])

    # Save preprocessed dataframe to Google Cloud Storage
    data_file_location = f"{pipeline_root}/{run_name}/prep_data.csv"
    scaled_df.to_csv(data_file_location, index=False)
    preprocessed_dataset.uri = data_file_location
    return


@component(
    # This component performs preprocessing on the default sklearn
    # iris dataset and to creates an Artifact of the preprocessed
    # dataset.
    packages_to_install=[
        "scikit-learn==1.2.1",
        "pandas==1.5.3",
        "google-cloud-storage==1.44.0",
        "gcsfs==2023.1.0",
        "fsspec==2023.1.0",
    ],
    base_image="python:3.10",
    output_component_file="./compiled_pipelines/train_model.yaml",
)
def train_model(
    preprocessed_dataset: Input[Dataset],
    model_artifact: Output[Model],
    metrics: Output[Metrics],
    pipeline_root: str,
    run_name: str,
) -> None:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from google.cloud import storage

    import pandas as pd
    import pickle

    scaled_df = pd.read_csv(f"{preprocessed_dataset.uri}")

    X_train, X_test, y_train, y_test = train_test_split(
        scaled_df.drop("target", axis=1),
        scaled_df["target"],
        test_size=0.33,
        random_state=42,
    )
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate and log accuracy
    accuracy = accuracy_score(y_test, y_pred, normalize=True)
    metrics.log_metric(metric="accuracy", value=accuracy)

    # Save model to gcp
    storage_client = storage.Client()
    file_location = f"{pipeline_root}/{run_name}/model.pkl"
    bucket_name = file_location.split("/")[2]
    file_gcs_path = file_location.replace(f"gs://{bucket_name}/", "")
    bucket = storage_client.bucket(bucket_name)

    # Write model to local temporary pickle file
    tmp_file_name = "tmp_model.pkl"
    with open(tmp_file_name, "wb") as file:
        pickle.dump(model, file)

    # Upload go Google Cloud Storage
    blob = bucket.blob(file_gcs_path)
    blob.upload_from_filename(tmp_file_name)

    # Update Model Artifact uri
    model_artifact.uri = file_location
    return
