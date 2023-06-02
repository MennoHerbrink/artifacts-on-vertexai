import kfp
from datetime import datetime
from kfp.v2 import compiler
from pipeline_components import preprocess_data, train_model
from google.cloud import aiplatform
from google.cloud.aiplatform import pipeline_jobs

project_id = "PROJECT_ID"
pipeline_root_path = "gs://GSC_PATH_ROOT"
region = "europe-west1"
current_time = datetime.now().strftime("%Y%m%d%H%M%S")
run_name = f"test-job-{current_time}"


# Define the workflow of the pipeline.
@kfp.dsl.pipeline(name="artifact-demo", pipeline_root=pipeline_root_path)
def pipeline(run_name: str):
    preprocess_data_step = preprocess_data(
        pipeline_root=pipeline_root_path, run_name=run_name
    ).set_display_name("Preprocess data")
    train_step = (
        train_model(
            preprocessed_dataset=preprocess_data_step.outputs["preprocessed_dataset"],
            pipeline_root=pipeline_root_path,
            run_name=run_name,
        )
        .set_display_name("Train model")
        .after(preprocess_data_step)
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
    parameter_values={"run_name": run_name},
    enable_caching=False,
    location="europe-west1",
)

job.submit()
