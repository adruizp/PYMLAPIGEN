import pytest
import pymlapigen, os
from pathlib import Path

resources = Path(__file__).parent / "resources"

@pytest.fixture()
def app():
    app = pymlapigen.flask_app
    app.config['TESTING'] = True
    app.config['APP_FOLDER'] = os.path.join(os.getcwd(), "test_files")

    yield app

@pytest.fixture()
def client(app):
    return app.test_client()

@pytest.fixture()
def classification_api():
    api = pymlapigen.api_generator.load_csv(resources / "iris.csv", separator=",")
    api.processNanNull("drop")
    api.setInputLabel("species")
    api.setAlgorithm("Classification","GNB")
    api.setAlgorithmParams({})
    api.setTestSize(0.3)
    api.setDropColumns([])
    api.step = 3
    api.ready = True
    api.trainModel()
    return api
