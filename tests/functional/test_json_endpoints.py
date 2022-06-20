from pathlib import Path
import json

resources = Path(__file__).parent.parent / "resources"

def test_home_endpoint(client):
    """
    GIVEN a Flask application configured for testing
    WHEN the '/api/' page is requested (GET)
    THEN check that the response is valid
    """
    response = client.get("/api/")
    assert b'{"status":"Api is working","generatedApis":{}}\n' in response.data

def test_api_endpoints(client):
    """
    GIVEN a Flask application configured for testing
    WHEN an API is generated
    THEN check that all API JSON endpoints are valid
    """

    with open(resources / "iris.json") as f:
        dataset = json.load(f)

    response = client.post("/api/load", json={
        "apiName": "API_TEST",    
        "inputLabel": "species",
        "modelType": "GNB",
        "dataset": dataset
    })

    assert b'{"success":"The API has been successfully generated and its now operable."' in response.data

    
    #API Endpoints
    response = client.get("/api/API_TEST")
    assert b'{"apiName":"API_TEST","mlProblem":"Multi-Label Classification","modelAlgorithm":"GaussianNB","label":"species"' in response.data
    
    response = client.get("/api/API_TEST/dataset")
    assert b'[{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2,"species":"Setosa"},{"sepal_length":4.9,' in response.data
    
    response = client.get("/api/API_TEST/metrics")
    assert b'{"accuracy":0.9777777777777777,"precision":0.9777777777777777,"recall":0.9743589743589745,"f1":0.974320987654321,"confu' in response.data
    
    response = client.get("/api/API_TEST/model")
    assert b'{"label":"species","features":["sepal_length","sepal_width","petal_length","petal_width"],"problem":"Classification"' in response.data