from pymlapigen import api_generator
import json
from pathlib import Path

resources = Path(__file__).parent.parent / "resources"


def test_load_csv():
    """
    GIVEN a csv file
    WHEN a new api is created
    THEN check the rows of the dataset, step and ready fields are defined correctly
    """
    api = api_generator.load_csv(resources / "iris.csv", separator=",")
    assert len(api.datasetDF) == 150
    assert api.step == 1
    assert api.ready == False


def test_load_json():
    """
    GIVEN a json dataset
    WHEN a new api is created
    THEN check the rows of the dataset, step and ready fields are defined correctly
    """
    with open(resources / "iris.json") as f:
        dataset = json.load(f)
    api = api_generator.load_json(dataset)
    assert len(api.datasetDF) == 150
    assert api.step == 1
    assert api.ready == False


def test_api_params(classification_api):
    """
    GIVEN a classification api
    WHEN a new api is generated
    THEN check the different attributes of the api
    """
    assert classification_api.inputLabel == "species"
    assert classification_api.getInputLabel() == "species"
    assert classification_api.modelType == "GNB"
    assert classification_api.getAlgorithm() == "GaussianNB"
    assert classification_api.mltype == "Classification"
    assert classification_api.getProblem() == "Multi-Label Classification"
    assert set(classification_api.getColumns()) == set(
        ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"])
    assert set(classification_api.getFeatures()) == set(
        ["sepal_length", "sepal_width", "petal_length", "petal_width"])
    assert classification_api.step == 3
    assert classification_api.ready == True
    assert classification_api.getModelParams() == {'label': 'species', 'features': ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'], 'problem': 'Classification', 'classification': 'Multi-Label', 'labels': [
        'Setosa', 'Versicolor', 'Virginica'], 'NanNull': 'drop', 'dropped': [], 'algorithm': 'GaussianNB', 'algorithm_args': {}, 'dataset_size': 150, 'training_size': 105, 'testing_size': 45}
