## My Changes from the Original Lab

### Dataset
- Replaced the Iris dataset (`sklearn.datasets.load_iris`) with the **[Palmer Penguins](https://allisonhorst.github.io/palmerpenguins/)** dataset (`data/penguins.csv`): 344 penguins of three species (Adelie, Chinstrap, Gentoo) from three islands in the Palmer Archipelago, Antarctica.
- Each penguin has four measurements (bill length, bill depth, flipper length, body mass) and two text columns (`island`, `sex`). The API now has to validate text inputs with a fixed set of allowed values, not only floats.
- `load_data` drops the 11 rows with missing values, leaving 333. `split_data` is now stratified, so the train and test sets keep the same share of each species.

### Model (`src/train.py`)
- Replaced the depth-3 `DecisionTreeClassifier` with a 200-tree `RandomForestClassifier`, inside a scikit-learn pipeline that one-hot encodes `island` and `sex`.
- The original split off a test set but never used it. `train.py` now prints a per-species classification report and the test accuracy: **0.990** (99 of 100 test penguins). The original depth-3 decision tree scores 0.94 to 0.97 on the same split.
- The saved file, `model/penguin_model.pkl`, holds the pipeline together with its details (species, test accuracy, feature importances, scikit-learn version). The API returns these details at `/model-info`.
- Removed `model/iris_model.pkl`.

### API (`src/main.py`, `src/predict.py`)
- Replaced `IrisData` with `PenguinData`:
  - `island` and `sex` are `Literal` types, so any other value is rejected with a 422.
  - The measurements have `Field(gt=0, le=...)` bounds, so negative or impossible values are rejected too.
  - An example request body is pre-filled under **Try it out** on `/docs`.
- Replaced `IrisResponse` (`response: int`, the class number) with `PenguinResponse`: the species name, the model's confidence, and the probability of each species.
- Added `GET /model-info` (with a `ModelInfo` response model) and `POST /predict/batch`, which accepts 1 to 100 penguins in one request.
- The model is now loaded once when the server starts (FastAPI's `lifespan`), not from disk on every request. If no model has been trained, `/` reports `"model_loaded": false` and the model endpoints return a 503 that says to run `train.py`, instead of a 500.
- The prediction endpoints are now plain `def` instead of `async def`. Model prediction is blocking CPU work, so inside `async def` it would stall every other request. FastAPI runs plain `def` endpoints in a thread pool instead.
- The model and data paths are now built from the code's location instead of `../model/...`, so the scripts work from any folder, not only from `src/`.
- Removed the docstrings and kept only the comments that explain something the code doesn't show.

### Requirements (`requirements.txt`)
- `scikit-learn==1.5.1` has no prebuilt wheels for Python 3.13 (they stop at 3.12), so on a current Python `pip install -r requirements.txt` has to compile scikit-learn from source, which needs a C compiler. Updated to `scikit-learn==1.9.1` and `pandas==3.1.0`, and replaced `fastapi[all]==0.111.1` with `fastapi[standard]==0.162.0`, the extra FastAPI now recommends (it includes uvicorn).
- Added `pytest==9.2.0` for the tests.

### Testing
- Added `tests/test_api.py`: eight tests using FastAPI's `TestClient`. They cover the health check, model info, single and batch predictions, and four kinds of invalid input that must return 422. `pytest.ini` puts `src/` on the import path. Run `pytest` from the `FastAPI_Labs` folder.

## Running the Lab Locally

### 1. Clone the repository
```bash
git clone https://github.com/perushw/MLOps.git
cd MLOps/Labs/API_Labs/FastAPI_Labs
```

### 2. Create a virtual environment and install the dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate it with `.venv\Scripts\activate` (PowerShell or Command Prompt) or `source .venv/Scripts/activate` (Git Bash). The repository's `.gitignore` already ignores `.venv`.

### 3. Train the model
```bash
cd src
python train.py
```

This prints the classification report and saves the model to `model/penguin_model.pkl`:

```plaintext
              precision    recall  f1-score   support

      Adelie       0.98      1.00      0.99        44
   Chinstrap       1.00      0.95      0.97        20
      Gentoo       1.00      1.00      1.00        36

    accuracy                           0.99       100

Test accuracy: 0.990
```

### 4. Start the API
From `src/`:

```bash
uvicorn main:app --reload
```

Open http://127.0.0.1:8000/docs to try the endpoints, or call them from a second terminal:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"island": "Biscoe", "sex": "female", "bill_length_mm": 46.1, "bill_depth_mm": 13.2, "flipper_length_mm": 211, "body_mass_g": 4500}'
```

```json
{"species":"Gentoo","confidence":1.0,"probabilities":{"Adelie":0.0,"Chinstrap":0.0,"Gentoo":1.0}}
```

### 5. Run the tests
The tests don't need the server running. From the `FastAPI_Labs` folder (where `pytest.ini` is):

```bash
pytest
```

```plaintext
8 passed
```


---
- Video Explanation: [FastAPI lab](https://www.youtube.com/watch?v=KReburHqRIQ&list=PLcS4TrUUc53LeKBIyXAaERFKBJ3dvc9GZ&index=4)
- Blog: [FastAPI Lab-1](https://www.mlwithramin.com/blog/fastapi-lab1)

---

## Overview

In this Lab, we will learn how to expose ML models as APIs using [FastAPI](https://fastapi.tiangolo.com/) and [uvicorn](https://www.uvicorn.org/).
1. **FastAPI**: FastAPI is a modern, fast (high-performance), web framework for building APIs with Python based on standard Python type hints.
2. **uvicorn**: Uvicorn is an [Asynchronous Server Gateway Interface - ASGI](https://youtu.be/vKjCkeJGbNk) web server implementation for Python. It is often used to serve FastAPI aplications.

The workflow involves the following steps:
1. Training a Decision Tree Classifier on Iris Dataset.
2. Serving the trained model as an API using FastAPI and uvicorn.

## Setting up the lab

1. Create a virtual environment(e.g. **fastapi_lab1_env**).
2. Activate the environment and install the required packages using `pip install -r requirements.txt`.

### Project structure

```
mlops_labs
└── fastapi_lab1
    ├── assets/
    ├── fastapi_lab1_env/
    ├── model/
    │   └── iris_model.pkl
    ├── src/
    │   ├── __init__.py
    │   ├── data.py
    │   ├── main.py
    │   ├── predict.py
    │   └── train.py
    ├── README.md
    └── requirements.txt
```

Note:
- **fastapi[all]** in **requirements.txt** will install optional additional dependencies for fastapi which contains **uvicorn** too.

## Running the Lab

1. First step is to train a Decision Tree Classifier(Although you have **`model/iris_model.pkl`** when you cloned from the repo, let's create a new model). To do this, move into **src/** folder with
    ```bash
    cd src
    ```
2. To train the Decision Tree Classifier, run:
    ```bash
    python train.py
    ```
3. To serve the trained model as an API, run:
    ```bash
    uvicorn app:main --reload
    ```
4. Testing endpoints - to view the documentation of your api model you can use [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) (or) [http://localhost:8000/docs](http://localhost:8000/docs) after you run you run your FastAPI app.
    
![API page](assets/docs.png)
   
You can also test out the results of your endpoints by interacting with them. Click on the dropdown button of your endpoint -> Try it out -> Fill the Request body -> Click on Execute button.

![API response](assets/api_response.png)

- You can also use other tools like [Postman](https://www.postman.com/) for API testing.

### FastAPI Syntax

- The instance of FASTAPI class can be defined as:
    ```bash
    app = FastAPI()
     ```
- When you run a FastAPI application, you often pass this app instance to an ASGI server uvicorn. The server then uses the app instance to handle incoming web requests and send responses based on the routes and logic you’ve defined in your FastAPI application.
- To run a FastAPI application, run:
    ```
    uvicorn main:app --reload
    ```
- In this command, **main** is the name of the Python file containing your app instance (without the .py extension), and **app** is the name of the instance itself. The **--reload** flag tells uvicorn to restart the server whenever code changes are detected, which is useful during development and should not be used in production.
- All the functions which should be used as API should be prefixed by **@app.get("/followed_by_endpoint_name")** or **@app.post("/followed_by_endpoint_name")**. This particular syntax is used to define route handlers (which function should handle an incoming request based on the URL and HTTP method), which are the functions responsible for responding to client requests to a given endpoint.
    1. **Decorator (@)**: This symbol is used to define a decorator, which is a way to dynamically add functionality to functions or methods. In FastAPI, decorators are used to associate a function with a particular HTTP method and path.
    2. **App Instance (app)**: This represents an instance of the FastAPI class. It is the core of your application and maintains the list of defined routes, request handlers, and other configurations.
    3. **HTTP Method (get, post, etc.)**: The HTTP method specifies the type of HTTP request the route will respond to. For example, get is used for retrieving data, and post is used for sending data to the server. FastAPI provides a decorator for each standard HTTP method, such as @app.put, @app.delete, @app.patch, and @app.options, allowing you to define handlers for different types of client requests. For detailed info refer to this webiste by [Mdn](https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods).
    4. **Path/Endpoint ("/endpoint_name")**: This is the URL path where the API will be accessible. When a client makes a request to this path using the specified HTTP method, FastAPI will execute the associated function and return the response.
- Using **async** in FastAPI allows for non-blocking operations, enabling the server to handle other requests while waiting for I/O tasks, like database queries or model loading, to complete. This leads to improved concurrency and resource utilization, enhancing the application's ability to manage multiple simultaneous requests efficiently.

### Data Models in FastAPI

##### 1. IrisData class

```python
class IrisData(BaseModel):
    petal_length: float
    sepal_length:float
    petal_width:float
    sepal_width:float
```

The **IrisData** class is a [Pydantic model](https://docs.pydantic.dev/latest/concepts/models/) which defines the expected structure of the data for a request body. When you use it as a type annotation for a route operation parameter, FastAPI will perform the following actions:
- **Request Body Reading:** FastAPI will read the request body as JSON.
- **Data Conversion:** It will convert the corresponding types, if necessary.
- **Data Validation:** It will validate the data. If the data is invalid, it will return a 422 Unprocessable Entity error response with details about what was incorrect.

#### 2. IrisResponse class

```python
class IrisResponse(BaseModel):
    response:int
```

The **IrisResponse** class is another Pydantic model that defines the structure of the response data for an endpoint. When you specify **response_model=IrisResponse** in a route operation, it tells FastAPI to:
- **Serialize the Output**: Convert the output data to JSON format according to the IrisResponse model.
- **Document the API**: Include the IrisResponse model in the generated API documentation, so API consumers know what to expect in the response.

---

### FastAPI features

1. **Request Body Reading**: When a client sends a request to a FastAPI endpoint, the request can include a body with data. For routes that expect data (commonly POST, PUT, or PATCH requests), this data is often in JSON format. FastAPI automatically reads the request body by checking the Content-Type header, which should be set to application/json for JSON payloads.
2. **Data Conversion**: Once the request body is read, FastAPI utilizes Pydantic models to parse the JSON data. Pydantic attempts to construct an instance of the specified model using the data from the request body. During this instantiation, Pydantic converts the JSON data into the proper Python data types as declared in the model.
    - For instance, if the JSON object has a field like petal_length with a value of "5.1" (a string), and the model expects a float, Pydantic will transform the string into a float. If conversion isn't possible (say, the value was "five point one"), Pydantic will raise a validation error.
3. **Data Validation**: Pydantic checks that all required fields are present and that the values are of the correct type, adhering to any constraints defined in the model (such as string length or number range). If the validation passes, the endpoint has a verified Python object to work with. If validation fails (due to missing fields, incorrect types, or constraint violations), FastAPI responds with a 422 Unprocessable Entity status. This response includes a JSON body detailing the validation errors, aiding clients in correcting their request data.
4. **Error Handling**: Error handling in FastAPI can be effectively managed using the HTTPException class. HTTPException is used to explicitly signal an HTTP error status code and return additional details about the error. When an HTTPException is raised within a route, FastAPI will catch the exception and use its content to form the HTTP response.
- **Instantiation**: The HTTPException class is instantiated with at least two arguments: status_code and detail. The status_code argument is an integer that represents the HTTP status code (e.g., 404 for Not Found, 400 for Bad Request). The detail argument is a string or any JSON-encodable object that describes the error.
- **Response**: When an HTTPException is raised, FastAPI sends an HTTP response with the status code specified. The detail provided in the HTTPException is sent as the body of the response in JSON format.

```python
from fastapi import FastAPI, HTTPException

app = FastAPI()

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    item = get_item_by_id(item_id)  # Hypothetical function to fetch an item
    if item is None:
        raise HTTPException(status_code=404, detail=f"Item with ID {item_id} not found")
    return item
```

In this example, **get_item_by_id** is a function that retrieves an item based on its ID. If no item with the given ID is found, an HTTPException with a 404 Not Found status code is raised, and the detail message is customized to include the ID of the item that was not found.

FastAPI will catch this exception and return a response with a 404 status code and a JSON body like this:

```json
{
    "detail": "Item with ID 1 not found"
}
```
- For more information on how to handle errors in FASTAPI refer to this [documentation](https://fastapi.tiangolo.com/tutorial/handling-errors/).
