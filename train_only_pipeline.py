import kfp
from datetime import datetime
from kfp.v2 import compiler, dsl
from pipeline_components import train_model
from google.cloud import aiplatform
from google.cloud.aiplatform import pipeline_jobs

project_id = "PROJECT_ID"
pipeline_root_path = "gs://GSC_PATH_ROOT"
data_file_location = "gs://GSC_PATH_DATA"
region = "europe-west1"
current_time = datetime.now().strftime("%Y%m%d%H%M%S")
run_name = f"test-job-{current_time}"


# Define the workflow of the pipeline.
@kfp.dsl.pipeline(name="artifact-demo", pipeline_root=pipeline_root_path)
def pipeline(run_name: str, data_file_location: str):
    import_dataset_step = dsl.importer(
        artifact_uri=data_file_location, artifact_class=dsl.Dataset, reimport=False
    ).set_display_name("Import preprocessed data")

    train_step = (
        train_model(
            preprocessed_dataset=import_dataset_step.output,
            pipeline_root=pipeline_root_path,
            run_name=run_name,
        )
        .set_display_name("Train model")
        .after(import_dataset_step)
    )


compiler.Compiler().compile(
    pipeline_func=pipeline, package_path="artifact-demo-vertex.json"
)

aiplatform.init(project=project_id, location=region)

job = pipeline_jobs.PipelineJob(
    project=project_id,
    display_name="vertex-artifacts",
    template_path="artifact-demo-vertex.json",
    job_id=run_name,
    pipeline_root=pipeline_root_path,
    parameter_values={
        "run_name": run_name,
        "data_file_location": data_file_location,
    },
    enable_caching=False,
    location="europe-west1",
)

job.submit()
