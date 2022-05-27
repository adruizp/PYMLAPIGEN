""" Aplicacion Flask """
from pymlapigen.api_generator import load_csv, load_json
from pymlapigen import flask_app

from flask import jsonify, render_template, request, redirect, url_for, send_file
from flask_mail import Mail, Message
from io import BytesIO
import os
import base64


""" Módulos locales """

mail = Mail(app=flask_app)

"""

    Rutas de la aplicación web.

"""

# Generated APIs
apis = {}

# Ruta HOME.

@flask_app.route("/")
def home():
    global apis
    return render_template("home.html", apis=apis)


# Ruta HOME (api).

@flask_app.route("/<apiName>")
def apiHome(apiName):
    global apis
    
    # Si apis[apiName] no esta definido, se retorna al paso 0
    if apiName not in apis:
        return redirect(url_for('get_load_0'))
        
    return render_template("home.html", apiName=apiName, api=apis[apiName], apis=apis)


# Ruta LOAD. Paso 0.


@flask_app.route("/load/0", methods=["GET"])
def get_load_0():
    global apis
    return render_template("load_0.html")


@flask_app.route("/load/0", methods=["POST"])
def post_load_0():

    global apis
    
    apiName = request.form['apiName']

    if apiName in apis:
        return render_template("load_0.html", error="There is already an API generated with that name. Try other name.")

    if apiName=="api":
        return render_template("load_0.html", error="The name of the Api cannot be \"api\". Try other name.")

    # Obtiene el separador del formulario
    separator = request.form['separator']

    # Obtiene el fichero (dataset) del formulario
    uploaded_file = request.files['file']

    # Comprueba que el fichero este subido correctamente
    if uploaded_file.filename != '':

        # Selecciona el path para alojar el fichero subido
        file_path = os.path.join(flask_app.config['APP_FOLDER'],
                                 flask_app.config['UPLOAD_FOLDER'],
                                 uploaded_file.filename)

        # Guarda el fichero subido en el path seleccionado
        uploaded_file.save(file_path)

    # Una vez subido el fichero, instancia un nuevo apis[apiName] usando como experimento el csv

    try:
        apis[apiName] = load_csv(file_path, separator=separator)

    except Exception as e:
        print(e)
        apis.pop(apiName, None)
        return render_template("load_0.html", error="An error has occurred reading the file.")

    # Redirecciona al siguiente paso
    return redirect(url_for('get_load_1', apiName=apiName))


# Ruta LOAD. Paso 1.
@flask_app.route("/<apiName>/load/1", methods=["GET"])
def get_load_1(apiName):

    global apis

    # Si apis[apiName] no esta definido, se retorna al paso 0
    if apiName not in apis:
        return redirect(url_for('get_load_0'))

    return render_template("load_1.html", apiName=apiName, api=apis[apiName], labels=apis[apiName].getColumns())


@flask_app.route("/<apiName>/load/1", methods=["POST"])
def post_load_1(apiName):

    global apis


    # Si apis[apiName] no esta definido, se retorna al paso 0
    if apiName not in apis:
        return redirect(url_for('get_load_0'))

    # El atributo ready de la API se marca a False ya que faltan pasos.
    apis[apiName].ready = False

    # Procesa los valores nulos y NaN
    apis[apiName].processNanNull(
        request.form['nan'], request.form['fillvalue'])

    # Fija el paso actual de la carga a 2. Esto habilitará iniciar el paso 2.
    apis[apiName].step = 2

    # Redirecciona al siguiente paso
    return redirect(url_for('get_load_2', apiName=apiName))


# Ruta LOAD. Paso 2.
@flask_app.route("/<apiName>/load/2", methods=["GET"])
def get_load_2(apiName):

    global apis

    # Si apis[apiName] no esta definido, se retorna al paso 0
    if apiName not in apis:
        return redirect(url_for('get_load_0'))

    # Si apis[apiName] se encuentra en el paso 1, se retorna al paso 1
    elif apis[apiName].step == 1:
        return redirect(url_for('get_load_1'))

    return render_template("load_2.html", apiName=apiName, api=apis[apiName], labels=apis[apiName].getColumns())


@flask_app.route("/<apiName>/load/2", methods=["POST"])
def post_load_2(apiName):

    global apis

    # Si apis[apiName] no esta definido, se retorna al paso 0
    if apiName not in apis:
        return redirect(url_for('get_load_0'))

    # Si apis[apiName] se encuentra en el paso 1, se retorna al paso 1
    elif apis[apiName].step == 1:
        return redirect(url_for('get_load_1'))

    # El atributo ready de la API se marca a False ya que faltan pasos.
    apis[apiName].ready = False

    # Obtiene la variable objetivo y la carga en la API
    inputLabel = request.form['inputLabel']
    apis[apiName].setInputLabel(inputLabel)

    # Obtiene el algoritmo para el modelo
    modelType = request.form['modelType']

    # A partir del algoritmo escogido, se obtiene el tipo de problema ML (Clasificación, Regresión o Clustering)
    classification = ["GNB", "SVC", "KNN", "DT", "RF"]
    regression = ["LR", "SVR", "SGDR", "KR", "GBR"]
    clustering = ["KM", "AP", "MS", "B"]

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

    # Fija el paso actual de la carga a 3. Esto habilitará iniciar el paso 3.
    apis[apiName].step = 3

    # Redirecciona al siguiente paso
    return redirect(url_for('get_load_3',apiName=apiName))

# Ruta LOAD. Paso 3.


@flask_app.route("/<apiName>/load/3", methods=["GET"])
def get_load_3(apiName):
    global apis

    # Si apis[apiName] no esta definido, se retorna al paso 0
    if apiName not in apis:
        return redirect(url_for('get_load_0'))

    # Si apis[apiName] se encuentra en el paso 1, se retorna al paso 1
    elif apis[apiName].step == 1:
        return redirect(url_for('get_load_1'))

    # Si apis[apiName] se encuentra en el paso 2, se retorna al paso 2
    elif apis[apiName].step == 2:
        return redirect(url_for('get_load_2'))

    # Si apis[apiName] esta definido y se encuentra en el paso 2, satisfactoriamente carga el paso 3
    return render_template("load_3.html", apiName=apiName, api=apis[apiName], problema=apis[apiName].getProblem(), algorithm=apis[apiName].getAlgorithm(), possibleLabels=apis[apiName].getPossibleLabels(), features=apis[apiName].getFeatures(), modelParams=apis[apiName].getAlgorithmParams())


@flask_app.route("/<apiName>/load/3", methods=["POST"])
def post_load_3(apiName):

    global apis

    # Si apis[apiName] no esta definido, se retorna al paso 0
    if apiName not in apis:
        return redirect(url_for('get_load_0'))

    # Si apis[apiName] se encuentra en el paso 1, se retorna al paso 1
    elif apis[apiName].step == 1:
        return redirect(url_for('get_load_1'))

    # Si apis[apiName] se encuentra en el paso 2, se retorna al paso 2
    elif apis[apiName].step == 2:
        return redirect(url_for('get_load_2'))

    # Se obtienen los parámetros por defecto del algoritmo
    algorithmParams = apis[apiName].getAlgorithmParams()

    # Se obtienen los parámetros escogidos en el formulario
    inputParams = request.form.getlist('modelParams')

    # Para cada parámetro del algoritmo
    for i, key in enumerate(algorithmParams):

        # Se obtiene el tipado del parámetro
        tipo = type(algorithmParams[key])

        # Se sobreescribe el parámetro por defecto por el introducido por el usuario (values)
        if(inputParams[i] != "None"):
            # Se realiza casting al parámetro introducido. Si no, todos serían del tipo str (String)

            if isinstance(algorithmParams[key], bool):
                castedInputParam = inputParams[i] == "True"
            else:
                castedInputParam = tipo(inputParams[i])

            algorithmParams[key] = castedInputParam

    # Carga los parámetros en la API
    apis[apiName].setAlgorithmParams(algorithmParams)

    # Obtiene las columnas a no tener en cuenta y las carga en la API
    dropColumns = request.form.getlist('dropColumns')
    apis[apiName].setDropColumns(dropColumns)

    # Obtiene el tamaño de test y lo carga en la API
    testSize = request.form['testSize']
    apis[apiName].setTestSize(testSize)

    # Si se trata de un problema de clasificación binario, obtiene el valor de la variable objetivo positivo
    # (para los True Positive, False Positive, True Negative y False Negative) y lo carga en la API
    if(apis[apiName].isBinaryClassification):
        positiveLabel = request.form['positiveLabel']
        apis[apiName].setPositiveLabel(positiveLabel)


    try:
        # Entrena el modelo
        apis[apiName].trainModel()

        # Evalúa el modelo
        apis[apiName].evaluateModel()

    except Exception as e:
        print(e)
        return render_template("load_3.html", apiName=apiName, api=apis[apiName], error=True, problema=apis[apiName].getProblem(), algorithm=apis[apiName].getAlgorithm(), possibleLabels=apis[apiName].getPossibleLabels(), features=apis[apiName].getFeatures(), modelParams=apis[apiName].getAlgorithmParams())


    # Si se llega a esta parte del código, significa que el entrenamiento y evaluación de modelo fue completo

    # Si el usuario marcó que desea recibir una notificación por correo electrónico y ha escrito su email,
    # envía un email al usuario. (Librería Flask-Mail)
    if(request.form['sendMail'] == 'Si') and ('email' in request.form):

        # Comprueba si el email no es un String vacio
        if request.form['email'] != "":

            # Email del usuario que recibira el correo
            email = request.form['email']

            # Parámetros del correo
            msg = Message('API generation complete', sender='tfgadrianruizparra@gmail.com', recipients=[email])
            msg.body = "The API has been generated successfully and its currently operable."

            # Envia el correo
            mail.send(msg)

    # El atributo ready de la API se marca a True.
    apis[apiName].ready = True

    return redirect(url_for('home'))

# Ruta DESTROY.
@flask_app.route("/destroy/<apiName>/")
def destroy(apiName):
    global apis
    del apis[apiName]
    return redirect(url_for('home'))



# Ruta DATASET.
@flask_app.route("/<apiName>/dataset")
def dataset(apiName):
    global apis

    # Si apis[apiName] (la api) no esta definido ni listo, se redirige a la carga de un nuevo dataset
    if apiName not in apis or not apis[apiName].ready:
        return redirect(url_for('get_load_0'))

    if "download" in request.args:
        if (request.args["download"] == "CSV"):

            return send_file(os.path.join(flask_app.config['APP_FOLDER'],
                                          flask_app.config['UPLOAD_FOLDER'],  download_CSV(apiName)))

    return render_template("dataset.html", apiName=apiName, api=apis[apiName], headers=apis[apiName].getColumns(), dataset=apis[apiName].getValues())


def download_CSV(apiName):
    global apis
    filename = apis[apiName].downloadCSV(os.path.join(flask_app.config['APP_FOLDER'],
                                                      flask_app.config['UPLOAD_FOLDER']))
    return filename

# Ruta METRICS.
@flask_app.route("/<apiName>/metrics")
def metrics(apiName):
    global apis

    # Si apis[apiName] (la api) no esta definido ni listo, se redirige a la carga de un nuevo dataset
    if apiName not in apis or not apis[apiName].ready:
        return redirect(url_for('get_load_0'))

    inputLabel = apis[apiName].getInputLabel()
    x_test, y_test, predictions = apis[apiName].getPredictions()

    return render_template("metrics.html", apiName=apiName, api=apis[apiName], headers=apis[apiName].metrics.keys(), metrics=apis[apiName].metrics.values(), test_headers = x_test.columns, test_label = inputLabel, x_test=x_test.values, y_test=y_test.values, predictions=predictions)


# Ruta MODEL.
@flask_app.route("/<apiName>/model")
def model(apiName):
    global apis

    # Si apis[apiName] (la api) no esta definido ni listo, se redirige a la carga de un nuevo dataset
    if apiName not in apis or not apis[apiName].ready:
        return redirect(url_for('get_load_0'))

    model = apis[apiName].getModelParams()
    return render_template("model.html", apiName=apiName, api=apis[apiName], headers=model.keys(), metrics=model.values())

# Ruta PREDICT.


@flask_app.route("/<apiName>/predict", methods=["GET"])
def predict(apiName):
    global apis

    # Si apis[apiName] (la api) no esta definido ni listo, se redirige a la carga de un nuevo dataset
    if apiName not in apis or not apis[apiName].ready:
        return redirect(url_for('get_load_0'))

    return render_template("predict.html", apiName=apiName, api=apis[apiName])


@flask_app.route("/<apiName>/predict", methods=["POST"])
def predict_post(apiName):
    global apis

    # Si apis[apiName] (la api) no esta definido ni listo, se redirige a la carga de un nuevo dataset
    if apiName not in apis or not apis[apiName].ready:
        return redirect(url_for('get_load_0'))

    # Acciones a realizar si se sube el formulario JSON
    if request.form['form'] == "JSON":
        jsonInput = request.form['jsonInput']
        try:
            resultPredictHeaders, resultPredictValues, typeResultHeaders, typeResultValues = apis[apiName].predictNewValues(
                jsonInput)
        except Exception as e:
            print(e)
            return render_template("predict.html", apiName=apiName, api=apis[apiName], error=True)

    # Acciones a realizar si se sube el formulario CSV
    else:
        # Guarda el fichero introducido similar a la carga de ficheros csv en path load
        csvInput = request.files['csvInput']
        if csvInput.filename != '':
            file_path = os.path.join(flask_app.config['APP_FOLDER'],
                                     flask_app.config['UPLOAD_FOLDER'], csvInput.filename)
            csvInput.save(file_path)

            try:
                # Predice los valores
                resultPredictHeaders, resultPredictValues, typeResultHeaders, typeResultValues = apis[apiName].predictNewValues(
                    file_path, typeData="CSV", separator=request.form['separator'])
            except Exception as e:
                print(e)
                return render_template("predict.html", apiName=apiName, api=apis[apiName], error=True)


    return render_template("predict.html", apiName=apiName, api=apis[apiName], headers=resultPredictHeaders, dataset=resultPredictValues, typeHeader=typeResultHeaders, typeDataset=typeResultValues)


# Ruta GRAPHS.
@flask_app.route("/<apiName>/graphs")
def graphs(apiName):
    global apis

    # Si apis[apiName] (la api) no esta definido ni listo, se redirige a la carga de un nuevo dataset
    if apiName not in apis or not apis[apiName].ready:
        return redirect(url_for('get_load_0'))

    # Generate the figure **without using pyplot**.
    figures = apis[apiName].graphs()
    data = []
    for fig in figures:
        # Save it to a temporary buffer.
        buf = BytesIO()
        fig.savefig(buf, format="png")
        # Embed the result in the html output.
        data.append(base64.b64encode(buf.getbuffer()).decode("ascii"))
    return render_template("graphs.html", apiName=apiName, api=apis[apiName], data=data)


"""

    Endpoints de la API REST.

    Todas estas rutas tienen el prefijo '/api/'

"""


@flask_app.route("/api/")
def defaultApiRoute():
    return jsonify({
        "status": "Api is working"
    })


@flask_app.route("/api/<apiName>/dataset")
def datasetApi(apiName):
    global apis

    if apiName not in apis or not apis[apiName].ready:
        return jsonify({"error": "API is not generated yet"})

    return jsonify(apis[apiName].filterDataset(request.args))


@flask_app.route("/api/<apiName>/metrics")
def metricsApi(apiName):
    global apis

    if apiName not in apis or not apis[apiName].ready:
        return jsonify({"error": "API is not generated yet"})

    return jsonify(apis[apiName].metrics)


@flask_app.route("/api/<apiName>/model")
def modelApi(apiName):
    global apis

    if apiName not in apis or not apis[apiName].ready:
        return jsonify({"error": "API is not generated yet"})

    return jsonify(apis[apiName].getModelParams())


@flask_app.route("/api/<apiName>/predict", methods=["POST"])
def predictApi(apiName):
    global apis

    if apiName not in apis or not apis[apiName].ready:
        return jsonify({"error": "API is not generated yet"})

    return jsonify(apis[apiName].predictNewValues(request.get_json(), toApi=True))


@flask_app.route("/api/<apiName>/load", methods=["POST"])
def loadApi(apiName):
    global apis

    data = request.get_json()

    # dataset, inputLabel y modelType son atributos que DEBEN incluir el cuerpo de la petición

    if "dataset" not in data:
        return jsonify({"error": "There is no dataset in the POST body and it must be supplied"})
    if "inputLabel" not in data:
        return jsonify({"error": "There is no inputLabel in the POST body and it must be supplied"})
    if "modelType" not in data:
        return jsonify({"error": "There is no modelType in the POST body and it must be supplied"})

    # Obtiene el dataset
    apis[apiName] = load_json(data["dataset"])

    # Procesa los NaN y los nulos
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

    # Obtiene el inputLabel
    apis[apiName].setInputLabel(data["inputLabel"])

    # Obtiene el algoritmo para el modelo
    modelType = data["modelType"]

    # A partir del algoritmo escogido, se obtiene el tipo de problema ML (Clasificación, Regresión o Clustering)
    classification = ["GNB", "SVC", "KNN", "DT", "RF"]
    regression = ["LR", "SVR", "SGDR", "KR", "GBR"]
    clustering = ["KM", "AP", "MS", "B"]

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

    # Carga los parámetros en la API si estan definidos en el cuerpo de la petición POST
    if "modelParams" in data:
        # Se obtienen los parámetros escogidos en el formulario
        apis[apiName].setAlgorithmParams(data["modelParams"])
    else:
        apis[apiName].setAlgorithmParams({})

    # Obtiene las columnas a no tener en cuenta y las carga en la API si estan definidos en el cuerpo de la petición POST
    if "dropColumns" in data:
        apis[apiName].setDropColumns(data["dropColumns"])
    else:
        apis[apiName].setDropColumns([])

     # Obtiene el tamaño de test y lo carga en la API si estan definidos en el cuerpo de la petición POST
    if "testSize" in data:
        apis[apiName].setTestSize(data["testSize"])
    else:
        apis[apiName].setTestSize(0.3)

    # Si se trata de un problema de clasificación binario, obtiene el valor de la variable objetivo positivo
    # (para los True Positive, False Positive, True Negative y False Negative) y lo carga en la API
    if apis[apiName].isBinaryClassification:
        if "positiveLabel" in data:
            apis[apiName].setPositiveLabel(data["positiveLabel"])
        else:
            return jsonify({"error": "positiveLabel must be suplied to Binary Classification problems"})

    apis[apiName].trainModel()
    apis[apiName].evaluateModel()

    if "email" in data:
        # Parámetros del correo
        msg = Message(
            'API generation complete', sender='tfgadrianruizparra@gmail.com', recipients=[data["email"]])
        msg.body = "The API has been generated successfully and its currently operable."
        # Envia el correo
        mail.send(msg)

    # El atributo ready de la API se marca a True.
    apis[apiName].ready = True

    return jsonify({"success": "The API has been successfully generated and its now operable."})
