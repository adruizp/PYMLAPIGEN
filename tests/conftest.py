import pytest
import pymlapigen, os


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
def runner(app):
    return app.test_cli_runner()
