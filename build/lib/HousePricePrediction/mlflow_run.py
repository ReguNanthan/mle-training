import mlflow
import mlflow.sklearn

from HousePricePrediction import ingest_data, train

remote_server_uri = "http://0.0.0.0:9000"  # set to your server URI
mlflow.set_tracking_uri(remote_server_uri)  # or set the MLFLOW_TRACKING_URI in the env

print(mlflow.tracking.get_tracking_uri())
exp_name = "House_Price_Prediction"
mlflow.set_experiment(exp_name)

# Create nested runs
# experiment_id = mlflow.create_experiment("experiment1")
with mlflow.start_run(
    run_name="PARENT_RUN",
    tags={"version": "v1", "priority": "P1"},
    description="parent",
) as parent_run:
    mlflow.log_param("parent", "yes")
    with mlflow.start_run(
        run_name="CHILD_RUN_INGEST",
        description="Ingest Data",
        nested=True,
    ) as child_run:
        mlflow.log_param("child1", "yes")
        ingest = ingest_data.main()
    with mlflow.start_run(
        run_name="CHILD_LINEAR_REG",
        description="Linear Regressor Model",
        nested=True,
    ) as child_run:
        mlflow.log_param("child2", "yes")
        output1 = train.main("Linear Regressor")
    with mlflow.start_run(
        run_name="CHILD_DECISON_TREE_REG",
        description="Decision Tree Regressor",
        nested=True,
    ) as child_run:
        mlflow.log_param("child3", "yes")
        output2 = train.main("Decision Tree Regressor")
    with mlflow.start_run(
        run_name="CHILD_RANDOM_RANDOMFOREST",
        description="RandomSearch RandomForest Regressor",
        nested=True,
    ) as child_run:
        mlflow.log_param("child3", "yes")
        output2 = train.main("RandomSearch RandomForest Regressor")
    with mlflow.start_run(
        run_name="CHILD_GRID_RANDAMFOREST",
        description="GridSearchCV RandomForest Regressor",
        nested=True,
    ) as child_run:
        mlflow.log_param("child3", "yes")
        output2 = train.main("GridSearchCV RandomForest Regressor")


print("parent run:")

print("run_id: {}".format(parent_run.info.run_id))
print("description: {}".format(parent_run.data.tags.get("mlflow.note.content")))
print("version tag value: {}".format(parent_run.data.tags.get("version")))
print("priority tag value: {}".format(parent_run.data.tags.get("priority")))
print("--")

# Search all child runs with a parent id

# query = "tags.mlflow.parentRunId = '{}'".format(parent_run.info.run_id)
# results = mlflow.search_runs(experiment_ids=[experiment_id], filter_string=query)
# print("child runs:")
# print(results[["run_id", "params.child", "tags.mlflow.runName"]])
