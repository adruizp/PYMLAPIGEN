from pathlib import Path

resources = Path(__file__).parent.parent / "resources"

def test_home_page(client):
    """
    GIVEN a Flask application configured for testing
    WHEN the '/' page is requested (GET)
    THEN check that the response is valid
    """
    response = client.get("/")
    assert b"<title>  Home </title>" in response.data

def test_import_page(client):
    """
    GIVEN a Flask application configured for testing
    WHEN the '/import' page is requested (GET)
    THEN check that the response is valid
    """
    response = client.get("/import")
    assert b"<title> Import API </title>" in response.data

def test_api_pages(client):
    """
    GIVEN a Flask application configured for testing
    WHEN an API is generated
    THEN check that all api pages are valid
    """

    #API Generation
    client.post("/load/0", data={
        "apiName": "API_TEST",
        "separator": ",",
        "file": (resources / "iris.csv").open("rb"),
    })

    client.post("/API_TEST/load/1", data={
        "nan": "drop",
        "fillvalue": "-1"
    })

    client.post("/API_TEST/load/2", data={
        "inputLabel": "species",
        "modelType": "GNB"
    })

    client.post("/API_TEST/load/3", data={
        "modelParams": ["None","1e-09"],
        "testSize": "0.3",
        "sendMail": "No"
    })

    #API Pages
    response = client.get("/API_TEST")
    assert b"<title>  API API_TEST </title>" in response.data

    response = client.get("/API_TEST/dataset")
    assert b"<title> Dataset - API API_TEST </title>" in response.data

    response = client.get("/API_TEST/metrics")
    assert b"<title> Metrics - API API_TEST </title>" in response.data

    response = client.get("/API_TEST/model")
    assert b"<title> Model - API API_TEST </title>" in response.data

    response = client.get("/API_TEST/predict")
    assert b"<title> Predict - API API_TEST </title>" in response.data

    response = client.get("/API_TEST/graphs")
    assert b"<title> Graphs - API API_TEST </title>" in response.data

