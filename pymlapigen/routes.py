# Imports
from re import A
from pymlapigen.api_generator import load_csv, load_json
from pymlapigen import flask_app

from flask import jsonify, render_template, request, redirect, url_for, send_file, abort
from flask_mail import Mail, Message
from io import BytesIO
import os
import base64
import pickle

# Local modules settings

mail = Mail(app=flask_app)

# Generated APIs
apis = {}


# Web app routes

@flask_app.route("/")
def home():
    """Home route."""

    global apis
    return render_template("home.html", apis=apis)


@flask_app.route("/<apiName>")
def apiHome(apiName):
    """API's home route

    Args:
        apiName (str): Name of the API
    """

    global apis
    # If api does not exists, redirect to generating a new one
    if apiName not in apis:
        return redirect(url_for('get_load_0'))

    # If apis[apiName] is on step 1, it will redirect to step 1
    elif apis[apiName].step == 1:
        return redirect(url_for('get_load_1', apiName=apiName))

    # If apis[apiName] is on step 2, it will redirect to step 2
    elif apis[apiName].step == 2:
        return redirect(url_for('get_load_2', apiName=apiName))

    # If apis[apiName] is on step 3, it will redirect to step 3
    elif apis[apiName].step == 3 and not apis[apiName].ready:
        return redirect(url_for('get_load_3', apiName=apiName))

    return render_template("home.html", apiName=apiName, api=apis[apiName], apis=apis, label=apis[apiName].getInputLabel(), problema=apis[apiName].getProblem(), algorithm=apis[apiName].getAlgorithm())



@flask_app.route("/load/0", methods=["GET"])
def get_load_0():
    """Load GET route. Step 0.

    Args:
        apiName (str): Name of the API 
    """
    global apis
    return render_template("load_0.html")


@flask_app.route("/load/0", methods=["POST"])
def post_load_0():
    """Load POST route. Step 0.

    Args:
        apiName (str): Name of the API 
    """

    global apis

    apiName = request.form['apiName']

    if apiName in apis:
        return render_template("load_0.html", error="There is already an API generated with that name. Try other name.")

    if apiName == "api":
        return render_template("load_0.html", error="The name of the Api cannot be \"api\". Try other name.")

    # Gets the form separator
    separator = request.form['separator']

    # Gets the form dataset file
    uploaded_file = request.files['file']

    # Checks if the file is uploaded successfully
    if uploaded_file.filename != '':

        # Gets the filepath
        file_path = os.path.join(flask_app.config['APP_FOLDER'],
                                 flask_app.config['UPLOAD_FOLDER'],
                                 uploaded_file.filename)

        # Saves the file in the filepath
        uploaded_file.save(file_path)

    # Once the file is saved, it creates an API instance using the file as dataset

    try:
        apis[apiName] = load_csv(file_path, separator=separator)

    except Exception as e:
        print(e)
        apis.pop(apiName, None)
        return render_template("load_0.html", error="An error has occurred reading the file.")

    # Redirects to the next step
    return redirect(url_for('get_load_1', apiName=apiName))


@flask_app.route("/<apiName>/load/1", methods=["GET"])
def get_load_1(apiName):
    """Load GET route. Step 1.

    Args:
        apiName (str): Name of the API 
    """

    global apis

    # If api does not exists, redirect to generating a new one
    if apiName not in apis:
        return redirect(url_for('get_load_0'))

    return render_template("load_1.html", apiName=apiName, api=apis[apiName], labels=apis[apiName].getColumns())


@flask_app.route("/<apiName>/load/1", methods=["POST"])
def post_load_1(apiName):
    """Load POST route. Step 1.

    Args:
        apiName (str): Name of the API 
    """

    global apis

    # If api does not exists, redirect to generating a new one
    if apiName not in apis:
        return redirect(url_for('get_load_0'))

    # API's ready attribute is set to False due to it needs training and evaluation
    apis[apiName].ready = False

    # Process NaN and Null values
    apis[apiName].processNanNull(
        request.form['nan'], request.form['fillvalue'])

    # Sets the current step to 2. This will enable step 2.
    apis[apiName].step = 2

    # Redirects to the next step
    return redirect(url_for('get_load_2', apiName=apiName))


@flask_app.route("/<apiName>/load/2", methods=["GET"])
def get_load_2(apiName):
    """Load GET route. Step 2.

    Args:
        apiName (str): Name of the API 
    """

    global apis

    # If api does not exists, redirect to generating a new one
    if apiName not in apis:
        return redirect(url_for('get_load_0'))

    # If apis[apiName] is on step 1, it will redirect to step 1
    elif apis[apiName].step == 1:
        return redirect(url_for('get_load_1'))

    return render_template("load_2.html", apiName=apiName, api=apis[apiName], labels=apis[apiName].getColumns())


@flask_app.route("/<apiName>/load/2", methods=["POST"])
def post_load_2(apiName):
    """Load POST route. Step 2.

    Args:
        apiName (str): Name of the API 
    """

    global apis

    # If api does not exists, redirect to generating a new one
    if apiName not in apis:
        return redirect(url_for('get_load_0'))

    # If apis[apiName] is on step 1, it will redirect to step 1
    elif apis[apiName].step == 1:
        return redirect(url_for('get_load_1'))

    # API's ready attribute is set to False due to it needs training and evaluation
    apis[apiName].ready = False

    # Gets the algoritmo para el modelo
    modelType = request.form['modelType']

    # A partir del algoritmo escogido, se obtiene el tipo de problema ML (Clasificación, Regresión o Clustering)
    classification = ["GNB", "SVC", "KNN", "DT", "RF"]
    regression = ["LR", "SVR", "SGDR", "KR", "GBR"]
    clustering = ["KM", "AP", "MS", "MKM"]

    if modelType in classification:
        mltype = "Classification"
    elif modelType in regression:
        mltype = "Regression"
    elif modelType in clustering:
        mltype = "Clustering"
    else:
        mltype = "Unknown"

    # Carga el tipo de problema ML y el algoritmo en la API
    apis[apiName].setAlgorithm(mltype, modelType)


    # Obtiene la variable objetivo y la carga en la API
    if mltype != "Clustering":
        inputLabel = request.form['inputLabel']
        apis[apiName].setInputLabel(inputLabel)


    # Sets the current step to 3. This will enable step 3.
    apis[apiName].step = 3

    # Redirects to the next step
    return redirect(url_for('get_load_3', apiName=apiName))


@flask_app.route("/<apiName>/load/3", methods=["GET"])
def get_load_3(apiName):
    """Load GET route. Step 3.

    Args:
        apiName (str): Name of the API 
    """

    global apis

    # If api does not exists, redirect to generating a new one
    if apiName not in apis:
        return redirect(url_for('get_load_0'))

    # If apis[apiName] is on step 1, it will redirect to step 1
    elif apis[apiName].step == 1:
        return redirect(url_for('get_load_1'))

    # If apis[apiName] is on step 2, it will redirect to step 2
    elif apis[apiName].step == 2:
        return redirect(url_for('get_load_2'))

    return render_template("load_3.html", apiName=apiName, api=apis[apiName], problema=apis[apiName].getProblem(), algorithm=apis[apiName].getAlgorithm(), label=apis[apiName].getInputLabel(), possibleLabels=apis[apiName].getPossibleLabels(), features=apis[apiName].getFeatures(), modelParams=apis[apiName].getAlgorithmParams())


@flask_app.route("/<apiName>/load/3", methods=["POST"])
def post_load_3(apiName):
    """Load POST route. Step 3.

    Args:
        apiName (str): Name of the API 
    """

    global apis

    # If api does not exists, redirect to generating a new one
    if apiName not in apis:
        return redirect(url_for('get_load_0'))

    # If apis[apiName] is on step 1, it will redirect to step 1
    elif apis[apiName].step == 1:
        return redirect(url_for('get_load_1'))

    # If apis[apiName] is on step 2, it will redirect to step 2
    elif apis[apiName].step == 2:
        return redirect(url_for('get_load_2'))

    # Algorithm default params
    algorithmParams = apis[apiName].getAlgorithmParams()

    # User input algorithm params
    inputParams = request.form.getlist('modelParams')

    # Sets the algorithm params for the API
    for i, key in enumerate(algorithmParams):

        tipo = type(algorithmParams[key])

        if(inputParams[i] != "None"):

            if isinstance(algorithmParams[key], bool):
                castedInputParam = inputParams[i] == "True"
            else:
                castedInputParam = tipo(inputParams[i])

            algorithmParams[key] = castedInputParam

    apis[apiName].setAlgorithmParams(algorithmParams)

    # Gets and sets the dropping columns of the API
    dropColumns = request.form.getlist('dropColumns')
    apis[apiName].setDropColumns(dropColumns)

    # Gets and sets the testSize of the API
    if apis[apiName].getProblem() != "Clustering":
        testSize = request.form['testSize']
        apis[apiName].setTestSize(testSize)

    # If it is an Binary Classification problem, it gets the positive Label
    # (in order to calculate False Negatives, True Negatives, False Positives, True Positives)
    if(apis[apiName].isBinaryClassification):
        positiveLabel = request.form['positiveLabel']
        apis[apiName].setPositiveLabel(positiveLabel)

    try:
        # Trains the model
        apis[apiName].trainModel()

        # Evaluates the model
        apis[apiName].evaluateModel()

    except Exception as e:
        print(e)
        return render_template("load_3.html", apiName=apiName, api=apis[apiName], error=True, problema=apis[apiName].getProblem(), algorithm=apis[apiName].getAlgorithm(), possibleLabels=apis[apiName].getPossibleLabels(), features=apis[apiName].getFeatures(), modelParams=apis[apiName].getAlgorithmParams())

    # If user checked the option of receiving an email, this code does the task
    # It sends an email to user email. (Module Flask-Mail)
    if ('sendMail' in request.form) and (request.form['sendMail'] == 'Si') and ('email' in request.form):

        # Check email is not an empty String
        if request.form['email'] != "":

            # User's email
            email = request.form['email']

            # Mail parameters
            msg = Message('API generation complete',
                          sender=flask_app.config["MAIL_USERNAME"], recipients=[email])
            msg.body = "The API has been generated successfully and its currently operable."

            # Send email
            mail.send(msg)

    # API generation is complete. Ready attribute is set to True.
    apis[apiName].ready = True

    return redirect(url_for('apiHome', apiName=apiName))


@flask_app.route("/destroy/<apiName>/")
def destroy(apiName):
    """Destroy route.

    Args:
        apiName (str): Name of the API to destroy
    """
    global apis
    del apis[apiName]
    return redirect(url_for('home'))


@flask_app.route("/<apiName>/dataset")
def dataset(apiName):
    """Dataset route.

    Args:
        apiName (str): Name of the API
    """

    global apis

    # If the api is not defined or ready, redirects to generating a new API
    if apiName not in apis or not apis[apiName].ready:
        return redirect(url_for('get_load_0'))

    # If user selects download option
    if "download" in request.args:
        if (request.args["download"] == "CSV"):
            print(os.path.join(flask_app.config['APP_FOLDER'],
                               flask_app.config['UPLOAD_FOLDER'],  download_CSV(apiName)))

            return send_file(os.path.join(flask_app.config['APP_FOLDER'],
                                          flask_app.config['UPLOAD_FOLDER'],  download_CSV(apiName)))

    return render_template("dataset.html", apiName=apiName, api=apis[apiName], headers=apis[apiName].getColumns(), dataset=apis[apiName].getValues())


# Generates a CSV file and returns its filename
def download_CSV(apiName):
    """Calls the download CSV function from the API and returns its filename

    Args:
        apiName (str): Name of the API
    """
    global apis
    filename = apis[apiName].downloadCSV(os.path.join(flask_app.config['APP_FOLDER'],
                                                      flask_app.config['UPLOAD_FOLDER']))
    return filename


@flask_app.route("/<apiName>/metrics")
def metrics(apiName):
    """Metrics route.

    Args:
        apiName (str): Name of the API
    """
    global apis

    # If the api is not defined or ready, redirects to generating a new API
    if apiName not in apis or not apis[apiName].ready:
        return redirect(url_for('get_load_0'))

    if apis[apiName].getProblem() != "Clustering":
        inputLabel = apis[apiName].getInputLabel()
        x_test, y_test, predictions = apis[apiName].getPredictions()
        return render_template("metrics.html", apiName=apiName, api=apis[apiName], problem=apis[apiName].getProblem(), headers=apis[apiName].metrics.keys(), metrics=apis[apiName].metrics.values(), test_headers=x_test.columns, test_label=inputLabel, x_test=x_test.values, y_test=y_test.values, predictions=predictions)
    else:
        x_test, predictions = apis[apiName].getPredictions()
        return render_template("metrics.html", apiName=apiName, api=apis[apiName], problem=apis[apiName].getProblem(), headers=apis[apiName].metrics.keys(), metrics=apis[apiName].metrics.values(), test_headers=x_test.columns, x_test=x_test.values, predictions=predictions)


# MODEL Route.
@flask_app.route("/<apiName>/model")
def model(apiName):
    """Model route.

    Args:
        apiName (str): Name of the API
    """
    global apis

    # If the api is not defined or ready, redirects to generating a new API
    if apiName not in apis or not apis[apiName].ready:
        return redirect(url_for('get_load_0'))

    model = apis[apiName].getModelParams()
    return render_template("model.html", apiName=apiName, api=apis[apiName], headers=model.keys(), metrics=model.values())

# PREDICT Route.


@flask_app.route("/<apiName>/predict", methods=["GET"])
def predict(apiName):
    """Predict GET route.

    Args:
        apiName (str): Name of the API
    """
    global apis

    # If the api is not defined or ready, redirects to generating a new API
    if apiName not in apis or not apis[apiName].ready:
        return redirect(url_for('get_load_0'))


    return render_template("predict.html", apiName=apiName, api=apis[apiName], features=apis[apiName].getFeatures().columns)


@flask_app.route("/<apiName>/predict", methods=["POST"])
def predict_post(apiName):
    """Predict POST route.

    Args:
        apiName (str): Name of the API
    """
    global apis

    # If the api is not defined or ready, redirects to generating a new API
    if apiName not in apis or not apis[apiName].ready:
        return redirect(url_for('get_load_0'))

    # Code for JSON form
    if request.form['form'] == "Input":
        args = request.form.copy()
        args.pop('form')
        
        try:
            # Predict the values
            resultPredictHeaders, resultPredictValues, typeResultHeaders, typeResultValues = apis[apiName].predictNewValues(
                args, typeData="Input")
        except Exception as e:
            print(e)
            return render_template("predict.html", apiName=apiName, api=apis[apiName], features=apis[apiName].getFeatures().columns, error=True)


    # Code for JSON form
    elif request.form['form'] == "JSON":
        jsonInput = request.form['jsonInput']
        try:
            resultPredictHeaders, resultPredictValues, typeResultHeaders, typeResultValues = apis[apiName].predictNewValues(
                jsonInput, typeData="JSON")
        except Exception as e:
            print(e)
            return render_template("predict.html", apiName=apiName, api=apis[apiName], features=apis[apiName].getFeatures().columns, error=True)

    # Code for CSV form
    else:
        # Saves the file
        csvInput = request.files['csvInput']
        if csvInput.filename != '':
            file_path = os.path.join(flask_app.config['APP_FOLDER'],
                                     flask_app.config['UPLOAD_FOLDER'], csvInput.filename)
            csvInput.save(file_path)

            try:
                # Predict the values
                resultPredictHeaders, resultPredictValues, typeResultHeaders, typeResultValues = apis[apiName].predictNewValues(
                    file_path, typeData="CSV", separator=request.form['separator'])
            except Exception as e:
                print(e)
                return render_template("predict.html", apiName=apiName, api=apis[apiName], features=apis[apiName].getFeatures().columns, error=True)

    return render_template("predict.html", apiName=apiName, api=apis[apiName], features=apis[apiName].getFeatures().columns, headers=resultPredictHeaders, dataset=resultPredictValues, typeHeader=typeResultHeaders, typeDataset=typeResultValues)


# Ruta GRAPHS.
@flask_app.route("/<apiName>/graphs")
def graphs(apiName):
    """Graphs route.

    Args:
        apiName (str): Name of the API
    """
    global apis

    # If the api is not defined or ready, redirects to generating a new API
    if apiName not in apis or not apis[apiName].ready:
        return redirect(url_for('get_load_0'))

    # Generates the figures.
    figures = apis[apiName].graphs()
    data = []
    for fig in figures:
        # Save every figure to a temporary buffer.
        buf = BytesIO()
        fig.savefig(buf, format="png")
        # Embed the result in the html output.
        data.append(base64.b64encode(buf.getbuffer()).decode("ascii"))
    return render_template("graphs.html", apiName=apiName, api=apis[apiName], data=data)


# EXPORT Route.
@flask_app.route("/<apiName>/export")
def export(apiName):
    """Export route.

    Args:
        apiName (str): Name of the API
    """
    global apis

    # If the api is not defined or ready, redirects to generating a new API
    if apiName not in apis or not apis[apiName].ready:
        return redirect(url_for('get_load_0'))

    # Generate file
    pickle.dump(apis[apiName],
                open(os.path.join(flask_app.config['APP_FOLDER'],
                                  flask_app.config['UPLOAD_FOLDER'], apiName+".api"), "wb"))

    # Download file
    return send_file(os.path.join(flask_app.config['APP_FOLDER'],
                                  flask_app.config['UPLOAD_FOLDER'],  apiName + ".api"))



@flask_app.route("/import", methods=["GET"])
def get_import():
    """Import GET route.

    Args:
        apiName (str): Name of the API
    """
    global apis
    return render_template("import.html")


@flask_app.route("/import", methods=["POST"])
def post_import():
    """Import POST route.

    Args:
        apiName (str): Name of the API
    """
    global apis

    apiName = request.form['apiName']

    if apiName in apis:
        return render_template("import.html", error="There is already an API generated with that name. Try other name.")

    if apiName == "api":
        return render_template("import.html", error="The name of the Api cannot be \"api\". Try other name.")

    # Gets the form import file
    uploaded_file = request.files['file']

    # Checks if the file is uploaded successfully
    if uploaded_file.filename != '':

        # Gets the filepath
        file_path = os.path.join(flask_app.config['APP_FOLDER'],
                                 flask_app.config['UPLOAD_FOLDER'],
                                 uploaded_file.filename)

        # Saves the file in the filepath
        uploaded_file.save(file_path)

    # Once the file is saved, it creates an API instance importing the file

    try:
        apis[apiName] = pickle.load(open(file_path, "rb"))

    except Exception as e:
        print(e)
        apis.pop(apiName, None)
        return render_template("import.html", error="An error has occurred reading the file.")

    # Redirects to API's home
    return redirect(url_for('apiHome', apiName=apiName))



# JSON API Endpoints.
# Every endpoint has the prefix '/api/'


@flask_app.route("/api/")
def defaultApiRoute():
    """JSON Home route.
    """
    return jsonify({
        "status": "Api is working"
    })


@flask_app.route("/api/<apiName>")
def homeApi(apiName):
    """JSON API Home route.

    Args:
        apiName (str): Name of the API
    """

    global apis

    if apiName not in apis or not apis[apiName].ready:
        return jsonify({"error": "API is not generated yet"})

    return jsonify({
        "apiName": apiName,
        "mlProblem": apis[apiName].getProblem(), 
        "modelAlgorithm": apis[apiName].getAlgorithm(),
        "label": apis[apiName].getInputLabel(),
        "endpoints": {
            "home": {"methods": "GET", "endpoint": url_for('homeApi', apiName=apiName)},
            "dataset": {"methods": "GET", "endpoint": url_for('datasetApi', apiName=apiName)},
            "metrics": {"methods": "GET", "endpoint": url_for('metricsApi', apiName=apiName)},
            "model": {"methods": "GET", "endpoint": url_for('modelApi', apiName=apiName)},
            "predict": {"methods": "POST", "endpoint": url_for('predictApi', apiName=apiName)},
        }})



@flask_app.route("/api/<apiName>/dataset")
def datasetApi(apiName):
    """JSON API Dataset route.

    Args:
        apiName (str): Name of the API
    """
    global apis

    if apiName not in apis or not apis[apiName].ready:
        return jsonify({"error": "API is not generated yet"})

    return jsonify(apis[apiName].filterDataset(request.args))


@flask_app.route("/api/<apiName>/metrics")
def metricsApi(apiName):
    """JSON API Metrics route.

    Args:
        apiName (str): Name of the API
    """
    global apis

    if apiName not in apis or not apis[apiName].ready:
        return jsonify({"error": "API is not generated yet"})

    return jsonify(apis[apiName].metrics)


@flask_app.route("/api/<apiName>/model")
def modelApi(apiName):
    """JSON API Model route.

    Args:
        apiName (str): Name of the API
    """
    global apis

    if apiName not in apis or not apis[apiName].ready:
        return jsonify({"error": "API is not generated yet"})

    return jsonify(apis[apiName].getModelParams())


@flask_app.route("/api/<apiName>/predict", methods=["POST"])
def predictApi(apiName):
    """JSON API Predict POST route.

    Args:
        apiName (str): Name of the API
    """
    global apis

    if apiName not in apis or not apis[apiName].ready:
        return jsonify({"error": "API is not generated yet"})

    return jsonify(apis[apiName].predictNewValues(request.get_json(), toApi=True))


@flask_app.route("/api/load", methods=["POST"])
def loadApi():
    """JSON API Load POST route.
    """
    global apis

    data = request.get_json()

    # apiName, dataset, inputLabel y modelType MUST BE in petition's body

    if "apiName" not in data:
        return jsonify({"error": "There is no apiName in the POST body and it must be supplied"})
    if "dataset" not in data:
        return jsonify({"error": "There is no dataset in the POST body and it must be supplied"})
    if "modelType" not in data:
        return jsonify({"error": "There is no modelType in the POST body and it must be supplied"})

    # Gets the API name
    apiName = data["apiName"]

    if apiName in apis:
        return jsonify({"error": "There is already an API generated with that name. Try other name."})

    if apiName == "api":
        return jsonify({"error": "The name of the Api cannot be \"api\". Try other name."})

    # Gets the dataset
    apis[apiName] = load_json(data["dataset"])

    # Process NaN and Null data
    if "nanNullMode" in data:
        if data["nanNullMode"] == "fill":
            if "fillvalue" in data:
                apis[apiName].processNanNull(
                    data["nanNullMode"], data["fillvalue"])
            else:
                return jsonify({"error": "NanNullMode is fill but no fillvalue is supplied"})
        else:
            apis[apiName].processNanNull(data["nanNullMode"])
    else:
        apis[apiName].processNanNull("drop")

    # Gets the model algorithm
    modelType = data["modelType"]

    # Gets the Machine Learning problem from the model algorithm (Classification, Regression or Clustering)
    classification = ["GNB", "SVC", "KNN", "DT", "RF"]
    regression = ["LR", "SVR", "SGDR", "KR", "GBR"]
    clustering = ["KM", "AP", "MS", "MKM"]

    if modelType in classification:
        mltype = "Classification"
    elif modelType in regression:
        mltype = "Regression"
    elif modelType in clustering:
        mltype = "Clustering"
    else:
        mltype = "Unknown"

    
    if mltype != "Clustering":
        if "inputLabel" not in data:
            return jsonify({"error": "There is no inputLabel in the POST body and it must be supplied"})
        else:        
            # Gets the inputLabel
            apis[apiName].setInputLabel(data["inputLabel"])

    # Sets the APIs model algorithm and ML problem
    apis[apiName].setAlgorithm(mltype, modelType)

    # Loads the algoritm parameters if they are defined on the POST's body
    if "modelParams" in data:
        apis[apiName].setAlgorithmParams(data["modelParams"])
    else:
        apis[apiName].setAlgorithmParams({})

    # Loads the dropping columns if they are defined on the POST's body
    if "dropColumns" in data:
        apis[apiName].setDropColumns(data["dropColumns"])
    else:
        apis[apiName].setDropColumns([])

     # Loads the test set size if they are defined on the POST's body. Defaults to 0.3
    if "testSize" in data:
        apis[apiName].setTestSize(data["testSize"])
    else:
        apis[apiName].setTestSize(0.3)

    # If it is an Binary Classification problem, it gets the positive Label
    # (in order to calculate False Negatives, True Negatives, False Positives, True Positives)
    if apis[apiName].isBinaryClassification:
        if "positiveLabel" in data:
            apis[apiName].setPositiveLabel(data["positiveLabel"])
        else:
            return jsonify({"error": "positiveLabel must be suplied to Binary Classification problems"})

    # Trains the model
    apis[apiName].trainModel()

    # Evaluates the model
    apis[apiName].evaluateModel()

    # If user did input his email, this code
    # sends an email. (Module Flask-Mail)
    if "email" in data:
        # Email parameters
        msg = Message(
            'API generation complete', sender='tfgadrianruizparra@gmail.com', recipients=[data["email"]])
        msg.body = "The API has been generated successfully and its currently operable."
        # Send email
        mail.send(msg)

    # API generation is complete. Ready attribute is set to True.
    apis[apiName].ready = True

    return jsonify({
        "success": "The API has been successfully generated and its now operable.",
        "endpoints": {
            "home": {"methods": "GET", "endpoint": url_for('homeApi', apiName=apiName)},
            "dataset": {"methods": "GET", "endpoint": url_for('datasetApi', apiName=apiName)},
            "metrics": {"methods": "GET", "endpoint": url_for('metricsApi', apiName=apiName)},
            "model": {"methods": "GET", "endpoint": url_for('modelApi', apiName=apiName)},
            "predict": {"methods": "POST", "endpoint": url_for('predictApi', apiName=apiName)},
        }})


"""

    Error handlers

"""

# Error 404


@flask_app.errorhandler(404)
def page_not_found(e):
    print(e)
    return render_template('error.html'), 404
