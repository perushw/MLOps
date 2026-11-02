## My Changes from the Original Lab

### Dataset
- Replaced the credit card customer data (`file.csv`, clustered on `BALANCE`, `PURCHASES` and `CREDIT_LIMIT`) with the **UCI Wine dataset** (`dags/data/wine.csv`).
- The Wine dataset has known labels (3 cultivars), so the clusters can be checked against reality. The `cultivar` column is dropped before training and only used for evaluation.
- Replaced the single-row `test.csv` with `dags/data/wine_test.csv`: 3 held-out wines, one from each cultivar.
- Removed the original `file.csv` and `test.csv`, since the pipeline no longer uses them.

### Model (`dags/src/lab.py`)
- Replaced K-Means + the elbow method (`kneed`) with a **Gaussian Mixture Model**, choosing `k` from 1 to 10 by the lowest **BIC**. `kneed` is no longer needed.
- Preprocessing now standardizes all 13 features (`StandardScaler`) and reduces them to 2 principal components (`PCA`), instead of min-max scaling 3 columns.
- Fixed model saving: the original saved the last model it fitted (`k=49`), not the one the elbow method picked. The saved model is now the lowest-BIC GMM.
- Fixed prediction: the original predicted on raw, unscaled `test.csv` rows with a model trained on scaled data. The preprocessor and GMM are now saved together as one scikit-learn pipeline, so new rows are scaled and projected the same way as the training data.
- Renamed `load_model_elbow` to `load_model_evaluate`. It prints the BIC for each `k`, scores the clusters against the true cultivars with the Adjusted Rand Index, and returns the cluster for every test row instead of only the first.
- Moved the file paths into `DATA_DIR` and `MODEL_DIR` constants and removed a leftover debug print.

### DAG (`dags/wine_clustering_dag.py`)
- Renamed `airflow.py` to `wine_clustering_dag.py`. The old name hid the real `airflow` library, so running the file directly failed with `ImportError: cannot import name 'DAG' from 'airflow'`.
- Changed the `PythonOperator` import back to `airflow.operators.python`, since `airflow.providers.standard` is not available in the Airflow 2.5.1 image used by `docker-compose.yaml`.
- Updated the DAG description, the model filename (`wine_gmm_model.sav`), and the last task to call `load_model_evaluate`.
- Removed the outdated XCom pickling note and trimmed the comments to the necessary ones.

### Testing
- Added `dags/src/run_pipeline.py`, a script that runs the four pipeline steps in order without Airflow.

## Running the Lab Locally

### 1. Clone the repository
```bash
git clone https://github.com/perushw/MLOps.git
cd MLOps/Labs/Airflow_Labs/Lab_1
```

### 2. Install the dependencies
```bash
pip install pandas scikit-learn
```

### 3. Run the pipeline without Airflow
```bash
python dags/src/run_pipeline.py
```

This runs the same four steps as the DAG. It prints the BIC for each `k`, the chosen number of clusters, the Adjusted Rand Index, and the cluster for each test wine, and saves the model to `dags/model/wine_gmm_model.sav`.

### 4. Run the DAG in Airflow
Set up `docker-compose.yaml` as described in [Airflow Setup](#airflow-setup), with `scikit-learn` added to the extra packages:

```yaml
_PIP_ADDITIONAL_REQUIREMENTS: ${_PIP_ADDITIONAL_REQUIREMENTS:- pandas scikit-learn}
```

Then start Airflow from `Labs/Airflow_Labs/Lab_1`:

```bash
docker compose up airflow-init
docker compose up
```

Open http://localhost:8080, log in (`airflow2` / `airflow2` by default), and trigger the `Airflow_Lab1` DAG. The `load_model_task` log shows the BIC scores, the Adjusted Rand Index and the test predictions.

To run the DAG once from the terminal instead, while the containers are up:

```bash
docker compose exec airflow-scheduler python dags/wine_clustering_dag.py
```

# Airflow lab

- In order to install Airflow using docker you can watch our [Airflow Lab1 Tutorial Video](https://youtu.be/exFSeGUbn4Q?feature=shared)
- For latest step-by-step instructions, check out this blog - [AirFlow Lab-1](https://www.mlwithramin.com/blog/airflow-lab1)

### ML Model

This lab clusters wines using a **Gaussian Mixture Model (GMM)** and picks the number of clusters with the **Bayesian Information Criterion (BIC)**. This replaces the original K-Means + elbow-method setup.

Unlike K-Means, which assumes round, equally sized clusters and gives each point one hard label, a GMM models every cluster as a Gaussian with its own (elliptical) covariance and gives each point a probability of belonging to each cluster. BIC is a likelihood score with a penalty for model complexity, so the best `k` is simply the one with the lowest BIC. No elbow-detection heuristic (`kneed`) is needed.

#### Dataset

[UCI Wine](https://archive.ics.uci.edu/dataset/109/wine) (also bundled with scikit-learn as `sklearn.datasets.load_wine`): 178 wines from three Italian cultivars, each described by 13 chemical measurements (alcohol, flavanoids, color intensity, proline, ...).

- `dags/data/wine.csv`: 175 rows used for clustering. The `cultivar` column is the true label. It is **dropped before training** and used only afterwards to check the clusters.
- `dags/data/wine_test.csv`: 3 held-out wines (one per cultivar, features only) that the trained model assigns to clusters.

#### Prerequisites

Before using this script, make sure you have the following libraries installed:

- pandas
- scikit-learn (sklearn)
- pickle

#### Usage

You can run the pipeline outside Airflow as follows (see `dags/src/run_pipeline.py`):

```python
# Load the data
data = load_data()

# Preprocess the data (standardize + PCA)
preprocessed_data = data_preprocessing(data)

# Fit GMMs for k = 1..10 and save the one with the lowest BIC
results = build_save_model(preprocessed_data, 'wine_gmm_model.sav')

# Load the saved model, report the chosen k, evaluate and predict
result = load_model_evaluate('wine_gmm_model.sav', results)
print(result)
```

Sample output:

```plaintext
Optimal no. of clusters (lowest BIC): 3
Adjusted Rand Index vs. true cultivars: 0.894
Test row 0: cluster 0 (probability 1.000)
Test row 1: cluster 2 (probability 1.000)
Test row 2: cluster 1 (probability 1.000)
```

BIC recovers the true number of cultivars (3), and the three held-out wines (one from each cultivar) land in three different clusters.

#### Functions

1. **load_data():**
   - *Description:* Loads `wine.csv`, serializes it, and returns the serialized data.
   - *Usage:*
     ```python
     data = load_data()
     ```

2. **data_preprocessing(data)**
   - *Description:* Deserializes data, drops missing rows and the `cultivar` label, standardizes the 13 features with `StandardScaler`, and projects them onto 2 principal components with `PCA`. Returns the reduced data together with the fitted preprocessor. Reducing to 2 dimensions keeps the GMM's full covariance matrices small enough to estimate reliably from ~175 rows.
   - *Usage:*
     ```python
     preprocessed_data = data_preprocessing(data)
     ```

3. **build_save_model(data, filename)**
   - *Description:* Fits a full-covariance `GaussianMixture` for each `k` from 1 to 10 and records each model's BIC. Saves the preprocessor and the lowest-BIC model as one scikit-learn pipeline (so it accepts raw feature rows), and returns the BIC scores and best `k`.
   - *Usage:*
     ```python
     results = build_save_model(preprocessed_data, 'wine_gmm_model.sav')
     ```

4. **load_model_evaluate(filename, results)**
   - *Description:* Loads the saved model and prints the BIC for each `k` and the selected number of clusters. It then scores the clusters against the true cultivars with the Adjusted Rand Index (1.0 = perfect match, 0 = random), and returns the predicted cluster for each row of `wine_test.csv`.
   - *Usage:*
     ```python
     result = load_model_evaluate('wine_gmm_model.sav', results)
     ```
### Airflow Setup

Use Airflow to author workflows as directed acyclic graphs (DAGs) of tasks. The Airflow scheduler executes your tasks on an array of workers while following the specified dependencies.

References

-   Product - https://airflow.apache.org/
-   Documentation - https://airflow.apache.org/docs/
-   Github - https://github.com/apache/airflow

#### Installation

Prerequisites: You should allocate at least 4GB memory for the Docker Engine (ideally 8GB).

Local

-   Docker Desktop Running

Cloud

-   Linux VM
-   SSH Connection
-   Installed Docker Engine - [Install using the convenience script](https://docs.docker.com/engine/install/ubuntu/#install-using-the-convenience-script)

#### Tutorial

1. Create a new directory

    ```bash
    mkdir -p ~/app
    cd ~/app
    ```

2. Running Airflow in Docker - [Refer](https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html#running-airflow-in-docker)

    a. You can check if you have enough memory by running this command

    ```bash
    docker run --rm "debian:bullseye-slim" bash -c 'numfmt --to iec $(echo $(($(getconf _PHYS_PAGES) * $(getconf PAGE_SIZE))))'
    ```

    b. Fetch [docker-compose.yaml](https://airflow.apache.org/docs/apache-airflow/2.5.1/docker-compose.yaml)

    ```bash
    curl -LfO 'https://airflow.apache.org/docs/apache-airflow/2.5.1/docker-compose.yaml'
    ```

    c. Setting the right Airflow user

    ```bash
    mkdir -p ./dags ./logs ./plugins ./working_data
    echo -e "AIRFLOW_UID=$(id -u)" > .env
    ```

    d. Update the following in docker-compose.yml

    ```bash
    # Donot load examples
    AIRFLOW__CORE__LOAD_EXAMPLES: 'false'

    # Additional python package
    _PIP_ADDITIONAL_REQUIREMENTS: ${_PIP_ADDITIONAL_REQUIREMENTS:- pandas }

    # Output dir
    - ${AIRFLOW_PROJ_DIR:-.}/working_data:/opt/airflow/working_data

    # Change default admin credentials
    _AIRFLOW_WWW_USER_USERNAME: ${_AIRFLOW_WWW_USER_USERNAME:-airflow2}
    _AIRFLOW_WWW_USER_PASSWORD: ${_AIRFLOW_WWW_USER_PASSWORD:-airflow2}
    ```

    e. Initialize the database

    ```bash
    docker compose up airflow-init
    ```

    f. Running Airflow

    ```bash
    docker compose up
    ```

    Wait until terminal outputs

    `app-airflow-webserver-1  | 127.0.0.1 - - [17/Feb/2023:09:34:29 +0000] "GET /health HTTP/1.1" 200 141 "-" "curl/7.74.0"`

    g. Enable port forwarding

    h. Visit `localhost:8080` login with credentials set on step `2.d`

3. Explore UI and add user `Security > List Users`

4. Create a python script [`dags/sandbox.py`](dags/sandbox.py)

    - BashOperator
    - PythonOperator
    - Task Dependencies
    - Params
    - Crontab schedules

    You can have n number of scripts inside dags dir

5. Stop docker containers

    ```bash
    docker compose down
    ```
### Airflow DAG Script

This Markdown file provides a detailed explanation of the Python script that defines an Airflow Directed Acyclic Graph (DAG) for a data processing and modeling workflow.

#### Script Overview

The script defines an Airflow DAG named `your_python_dag` that consists of several tasks. Each task represents a specific operation in a data processing and modeling workflow. The script imports necessary libraries, sets default arguments for the DAG, creates PythonOperators for each task, defines task dependencies, and provides command-line interaction with the DAG.

#### Importing Libraries

```python
# Import necessary libraries and modules
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from src.lab import load_data, data_preprocessing, build_save_model, load_model_evaluate
from airflow import configuration as conf
```
The script starts by importing the required libraries and modules. Notable imports include the `DAG` and `PythonOperator` classes from the `airflow` package, datetime manipulation functions, and custom functions from the `src.lab` module.



#### Enable pickle support for XCom, allowing data to be passed between tasks
```python
conf.set('core', 'enable_xcom_pickling', 'True')
```

#### Define default arguments for your DAG
```python
default_args = {
    'owner': 'your_name',
    'start_date': datetime(2023, 9, 17),
    'retries': 0,  # Number of retries in case of task failure
    'retry_delay': timedelta(minutes=5),  # Delay before retries
}
```
Default arguments for the DAG are specified in a dictionary named default_args. These arguments include the DAG owner's name, the start date, the number of retries, and the retry delay in case of task failure.

#### Create a DAG instance named 'your_python_dag' with the defined default arguments
``` python 
dag = DAG(
    'your_python_dag',
    default_args=default_args,
    description='Your Python DAG Description',
    schedule_interval=None,  # Set the schedule interval or use None for manual triggering
    catchup=False,
)
```
Here, the DAG object dag is created with the name 'your_python_dag' and the specified default arguments. The description provides a brief description of the DAG, and schedule_interval defines the execution schedule (in this case, it's set to None for manual triggering). catchup is set to False to prevent backfilling of missed runs.


#### Task to load data, calls the 'load_data' Python function
``` python 
load_data_task = PythonOperator(
    task_id='load_data_task',
    python_callable=load_data,
    dag=dag,
)
```

#### Task to perform data preprocessing, depends on 'load_data_task'
```python 
data_preprocessing_task = PythonOperator(
    task_id='data_preprocessing_task',
    python_callable=data_preprocessing,
    op_args=[load_data_task.output],
    dag=dag,
)
```
The 'data_preprocessing_task' depends on the 'load_data_task' and calls the data_preprocessing function, which is provided with the output of the 'load_data_task'.

#### Task to build and save a model, depends on 'data_preprocessing_task'
```python
build_save_model_task = PythonOperator(
    task_id='build_save_model_task',
    python_callable=build_save_model,
    op_args=[data_preprocessing_task.output, "wine_gmm_model.sav"],
    provide_context=True,
    dag=dag,
)
```
The 'build_save_model_task' depends on the 'data_preprocessing_task' and calls the build_save_model function. It also provides additional context information and arguments.

#### Task to load a model using the 'load_model_evaluate' function, depends on 'build_save_model_task'
``` python
load_model_task = PythonOperator(
    task_id='load_model_task',
    python_callable=load_model_evaluate,
    op_args=["wine_gmm_model.sav", build_save_model_task.output],
    dag=dag,
)
```
The 'load_model_task' depends on the 'build_save_model_task' and calls the load_model_evaluate function with specific arguments.

#### Set task dependencies
```python
load_data_task >> data_preprocessing_task >> build_save_model_task >> load_model_task
```
Task dependencies are defined using the >> operator. In this case, the tasks are executed in sequence: 'load_data_task' -> 'data_preprocessing_task' -> 'build_save_model_task' -> 'load_model_task'.

#### If this script is run directly, allow command-line interaction with the DAG
```python
if __name__ == "__main__":
    dag.cli()
```
- Lastly, the script allows for command-line interaction with the DAG. When the script is run directly, the dag.cli() function is called, providing the ability to trigger and manage the DAG from the command line.
- This script defines a comprehensive Airflow DAG for a data processing and modeling workflow, with clear task dependencies and default arguments.

### Running an Apache Airflow DAG Pipeline in Docker

This guide provides detailed steps to set up and run an Apache Airflow Directed Acyclic Graph (DAG) pipeline within a Docker container using Docker Compose. The pipeline is named "your_python_dag."

#### Prerequisites

- Docker: Make sure Docker is installed and running on your system.

#### Step 1: Directory Structure

Ensure your project has the following directory structure:

```plaintext
Lab_1/
├── dags/
│   ├── wine_clustering_dag.py   # DAG definition
│   ├── src/
│   │   ├── lab.py               # Data processing and modeling functions
│   │   └── run_pipeline.py      # Runs the pipeline without Airflow
│   ├── data/                    # wine.csv and wine_test.csv
│   └── model/                   # Saved model, created by the pipeline
├── docker-compose.yaml          # Docker Compose configuration
```

#### Step 2: Docker Compose Configuration
Create a docker-compose.yaml file in the project root directory. This file defines the services and configurations for running Airflow in a Docker container.

#### Step 3: Start the Docker containers by running the following command

```plaintext
docker compose up
```

Wait until you see the log message indicating that the Airflow webserver is running:

```plaintext
app-airflow-webserver-1 | 127.0.0.1 - - [17/Feb/2023:09:34:29 +0000] "GET /health HTTP/1.1" 200 141 "-" "curl/7.74.0"
```

#### Step 4: Access Airflow Web Interface
- Open a web browser and navigate to http://localhost:8080.

- Log in with the credentials set in the .env file or use the default credentials (username: admin, password: admin).

- Once logged in, you'll be on the Airflow web interface.

#### Step 5: Trigger the DAG
- In the Airflow web interface, navigate to the "DAGs" page.

- You should see the "your_python_dag" listed.

- To manually trigger the DAG, click on the "Trigger DAG" button or enable the DAG by toggling the switch to the "On" position.

- Monitor the progress of the DAG in the Airflow web interface. You can view logs, task status, and task execution details.

#### Step 6: Pipeline Outputs

- Once the DAG completes its execution, check any output or artifacts produced by your functions and tasks. 