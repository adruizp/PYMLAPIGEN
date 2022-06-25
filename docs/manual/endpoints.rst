=========
Endpoints
=========

En esta sección se comentarán los distintos endpoints para peticiones HTTP (GET y POST) de la aplicación.


Home (GET)
----------
http://localhost:5000/api/

Este endpoint comprueba que la aplicación esté en **ejecución** y devuelve **todas las APIs** generadas y accesibles en memoria.

::

    {
        "status": "Api is working",
        "generatedApis": {
            "ClassificationAPIFromJSON": "/api/ClassificationAPIFromJSON",
            "RegressionAPIFromJSON": "/api/RegressionAPIFromJSON"
            "ClusteringAPIFromJSON": "/api/ClusteringAPIFromJSON",
        }
    }

Load (POST)
-----------
http://localhost:5000/api/load

Este endpoint **genera una nueva API**. Para ello, utiliza el cuerpo provisto en la **petición HTTP POST**.

Para más información acerca de esta petición, consultar la sección :ref:`endpoint-json-post` de :doc:`generation`.


API Home (GET)
--------------
http://localhost:5000/api/NOMBREAPI

Este endpoint devuelve los parámetros principales de la API junto a sus endpoints.

::

    {
        "apiName": "ClassificationAPIFromJSON",
        "mlProblem": "Multi-Label Classification",
        "modelAlgorithm": "GaussianNB",
        "label": "species",
        "endpoints": {
            "home": {
                "methods": "GET",
                "endpoint": "/api/ClassificationAPIFromJSON"
            },
            "dataset": {
                "methods": "GET",
                "endpoint": "/api/ClassificationAPIFromJSON/dataset"
            },
            "metrics": {
                "methods": "GET",
                "endpoint": "/api/ClassificationAPIFromJSON/metrics"
            },
            "model": {
                "methods": "GET",
                "endpoint": "/api/ClassificationAPIFromJSON/model"
            },
            "predict": {
                "methods": "POST",
                "endpoint": "/api/ClassificationAPIFromJSON/predict"
            }
        }
    }



API Dataset (GET)
-----------------
http://localhost:5000/api/NOMBREAPI/dataset

Este endpoint devuelve el **dataset**. Además, es posible especificar **filtros** como parámetros en la URL de la forma "atributo=valor".

Un ejemplo de filtro puede ser /api/ClassificationAPIFromJSON/dataset?species=Setosa. En este ejemplo, el filtro **?species=Setosa** implica que únicamente se devuelva aquellas filas cuya especie sea Setosa.

::

    [
        {
            "petal_length": 1.4,
            "petal_width": 0.2,
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "species": "Setosa"
        },
        {
            "petal_length": 1.4,
            "petal_width": 0.2,
            "sepal_length": 4.9,
            "sepal_width": 3.0,
            "species": "Setosa"
        },
        ...
    ]



API Metrics (GET)
-----------------
http://localhost:5000/api/NOMBREAPI/metrics

Este endpoint **retorna** las métricas del experimento de la API generada.

::

    {
        "accuracy": 0.9777777777777777,
        "precision": 0.9777777777777777,
        "recall": 0.9743589743589745,
        "f1": 0.974320987654321,
        "confusion_matrix": [
            [
                19,
                0,
                0
            ],
            [
                0,
                12,
                1
            ],
            [
                0,
                0,
                13
            ]
        ]
    }


API Model (GET)
---------------
http://localhost:5000/api/NOMBREAPI/model

Este endpoint **retorna** los parámetros escogidos para el experimento de la API generada.

::

    {
        "label": "species",
        "features": [
            "petal_length",
            "petal_width",
            "sepal_length",
            "sepal_width"
        ],
        "problem": "Classification",
        "classification": "Multi-Label",
        "labels": [
            "Setosa",
            "Versicolor",
            "Virginica"
        ],
        "NanNull": "drop",
        "dropped": [],
        "algorithm": "GaussianNB",
        "algorithm_args": {},
        "dataset_size": 150,
        "training_size": 105,
        "testing_size": 45
    }

API Predict (POST)
------------------
http://localhost:5000/api/NOMBREAPI/predict

Este endpoint permite **realizar** predicciones utilizando nuevos datos. Estos nuevos deben estar incluidos en el **cuerpo de la petición POST**.

Por ejemplo, un cuerpo podría ser:

::

    [{
      "sepal_length": 4.9,
      "sepal_width": 3.0,
      "petal_length": 1.4,
      "petal_width": 0.2
   }, {
      "sepal_length": 6.3,
      "sepal_width": 2.7,
      "petal_length": 4.9,
      "petal_width": 1.8
   }, {
      "sepal_length": 4.8,
      "sepal_width": 3.1,
      "petal_length": 1.6,
      "petal_width": 0.2
   }]

El resultado de la petición es:

   - El **valor predicho** para cada entrada en experimentos de **clasificación** y **regresión**.
   - El **clúster** al que pertenece cada entrada en experimentos de **clustering**.

::

   [
    "Setosa",
    "Virginica",
    "Setosa"
   ]