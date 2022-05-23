""" Aplicacion Flask """
from tkinter.messagebox import NO
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

api_generator = None

# Ruta HOME.


@flask_app.route("/")
def home():
    global api_generator
    return render_template("home.html", api=api_generator)

# Ruta LOAD. Paso 0.


@flask_app.route("/load/0", methods=["GET"])
def get_load_0():
    global api_generator
    return render_template("load_0.html", api=api_generator)


@flask_app.route("/load/0", methods=["POST"])
def post_load_0():

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

    # Una vez subido el fichero, instancia un nuevo api_generator usando como experimento el csv
    global api_generator

    try:
        api_generator = load_csv(file_path, separator=separator)

    except Exception as e:
        print(e)
        api_generator = None
        return render_template("load_0.html", api=api_generator, error=True)

    # Redirecciona al siguiente paso
    return redirect(url_for('get_load_1'))


# Ruta LOAD. Paso 1.
@flask_app.route("/load/1", methods=["GET"])
def get_load_1():

    global api_generator

    # Si api_generator no esta definido, se retorna al paso 0
    if api_generator is None:
        return redirect(url_for('get_load_0'))

    return render_template("load_1.html", api=api_generator, labels=api_generator.getColumns())


@flask_app.route("/load/1", methods=["POST"])
def post_load_1():

    global api_generator

    # Si api_generator no esta definido, se retorna al paso 0
    if api_generator is None:
        return redirect(url_for('get_load_0'))

    # El atributo ready de la API se marca a False ya que faltan pasos.
    api_generator.ready = False

    # Procesa los valores nulos y NaN
    api_generator.processNanNull(
        request.form['nan'], request.form['fillvalue'])

    # Fija el paso actual de la carga a 2. Esto habilitará iniciar el paso 2.
    api_generator.step = 2

    # Redirecciona al siguiente paso
    return redirect(url_for('get_load_2'))


# Ruta LOAD. Paso 2.
@flask_app.route("/load/2", methods=["GET"])
def get_load_2():

    global api_generator

    # Si api_generator no esta definido, se retorna al paso 0
    if api_generator is None:
        return redirect(url_for('get_load_0'))

    # Si api_generator se encuentra en el paso 1, se retorna al paso 1
    elif api_generator.step == 1:
        return redirect(url_for('get_load_1'))

    return render_template("load_2.html", api=api_generator, labels=api_generator.getColumns())


@flask_app.route("/load/2", methods=["POST"])
def post_load_2():

    global api_generator

    # Si api_generator no esta definido, se retorna al paso 0
    if api_generator is None:
        return redirect(url_for('get_load_0'))

    # Si api_generator se encuentra en el paso 1, se retorna al paso 1
    elif api_generator.step == 1:
        return redirect(url_for('get_load_1'))

    # El atributo ready de la API se marca a False ya que faltan pasos.
    api_generator.ready = False

    # Obtiene la variable objetivo y la carga en la API
    inputLabel = request.form['inputLabel']
    api_generator.setInputLabel(inputLabel)

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
    api_generator.setAlgorithm(mltype, modelType)

    # Fija el paso actual de la carga a 3. Esto habilitará iniciar el paso 3.
    api_generator.step = 3

    # Redirecciona al siguiente paso
    return redirect(url_for('get_load_3'))

# Ruta LOAD. Paso 3.


@flask_app.route("/load/3", methods=["GET"])
def get_load_3():
    global api_generator

    # Si api_generator no esta definido, se retorna al paso 0
    if api_generator is None:
        return redirect(url_for('get_load_0'))

    # Si api_generator se encuentra en el paso 1, se retorna al paso 1
    elif api_generator.step == 1:
        return redirect(url_for('get_load_1'))

    # Si api_generator se encuentra en el paso 2, se retorna al paso 2
    elif api_generator.step == 2:
        return redirect(url_for('get_load_2'))

    # Si api_generator esta definido y se encuentra en el paso 2, satisfactoriamente carga el paso 3
    return render_template("load_3.html", api=api_generator, problema=api_generator.getProblem(), algorithm=api_generator.getAlgorithm(), possibleLabels=api_generator.getPossibleLabels(), features=api_generator.getFeatures(), modelParams=api_generator.getAlgorithmParams())


@flask_app.route("/load/3", methods=["POST"])
def post_load_3():

    global api_generator

    # Si api_generator no esta definido, se retorna al paso 0
    if api_generator is None:
        return redirect(url_for('get_load_0'))

    # Si api_generator se encuentra en el paso 1, se retorna al paso 1
    elif api_generator.step == 1:
        return redirect(url_for('get_load_1'))

    # Si api_generator se encuentra en el paso 2, se retorna al paso 2
    elif api_generator.step == 2:
        return redirect(url_for('get_load_2'))

    # Se obtienen los parámetros por defecto del algoritmo
    algorithmParams = api_generator.getAlgorithmParams()

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
    api_generator.setAlgorithmParams(algorithmParams)

    # Obtiene las columnas a no tener en cuenta y las carga en la API
    dropColumns = request.form.getlist('dropColumns')
    api_generator.setDropColumns(dropColumns)

    # Obtiene el tamaño de test y lo carga en la API
    testSize = request.form['testSize']
    api_generator.setTestSize(testSize)

    # Si se trata de un problema de clasificación binario, obtiene el valor de la variable objetivo positivo
    # (para los True Positive, False Positive, True Negative y False Negative) y lo carga en la API
    if(api_generator.isBinaryClassification):
        positiveLabel = request.form['positiveLabel']
        api_generator.setPositiveLabel(positiveLabel)


    try:
        # Entrena el modelo
        api_generator.trainModel()

        # Evalúa el modelo
        api_generator.evaluateModel()
    except Exception as e:
        print(e)
        return render_template("load_3.html", api=api_generator, error=True, problema=api_generator.getProblem(), algorithm=api_generator.getAlgorithm(), possibleLabels=api_generator.getPossibleLabels(), features=api_generator.getFeatures(), modelParams=api_generator.getAlgorithmParams())


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
    api_generator.ready = True

    return redirect(url_for('home'))

# Ruta DESTROY.
@flask_app.route("/destroy")
def destroy():
    global api_generator
    del api_generator
    api_generator = None
    return redirect(url_for('home'))



# Ruta DATASET.
@flask_app.route("/dataset")
def dataset():
    global api_generator

    # Si api_generator (la api) no esta definido ni listo, se redirige a la carga de un nuevo dataset
    if api_generator is None or not api_generator.ready:
        return redirect(url_for('get_load_0'))

    if "download" in request.args:
        if (request.args["download"] == "CSV"):

            return send_file(os.path.join(flask_app.config['APP_FOLDER'],
                                          flask_app.config['UPLOAD_FOLDER'],  download_CSV()))

    return render_template("dataset.html", api=api_generator, headers=api_generator.getColumns(), dataset=api_generator.getValues())


def download_CSV():
    global api_generator
    filename = api_generator.downloadCSV(os.path.join(flask_app.config['APP_FOLDER'],
                                                      flask_app.config['UPLOAD_FOLDER']))
    return filename

# Ruta METRICS.


@flask_app.route("/metrics")
def metrics():
    global api_generator

    # Si api_generator (la api) no esta definido ni listo, se redirige a la carga de un nuevo dataset
    if api_generator is None or not api_generator.ready:
        return redirect(url_for('get_load_0'))

    inputLabel = api_generator.getInputLabel()
    x_test, y_test, predictions = api_generator.getPredictions()

    return render_template("metrics.html", api=api_generator, headers=api_generator.metrics.keys(), metrics=api_generator.metrics.values(), test_headers = x_test.columns, test_label = inputLabel, x_test=x_test.values, y_test=y_test.values, predictions=predictions)


# Ruta MODEL.
@flask_app.route("/model")
def model():
    global api_generator

    # Si api_generator (la api) no esta definido ni listo, se redirige a la carga de un nuevo dataset
    if api_generator is None or not api_generator.ready:
        return redirect(url_for('get_load_0'))

    model = api_generator.getModelParams()
    return render_template("model.html", api=api_generator, headers=model.keys(), metrics=model.values())

# Ruta PREDICT.


@flask_app.route("/predict", methods=["GET"])
def predict():
    global api_generator

    # Si api_generator (la api) no esta definido ni listo, se redirige a la carga de un nuevo dataset
    if api_generator is None or not api_generator.ready:
        return redirect(url_for('get_load_0'))

    return render_template("predict.html", api=api_generator)


@flask_app.route("/predict", methods=["POST"])
def predict_post():
    global api_generator

    # Si api_generator (la api) no esta definido ni listo, se redirige a la carga de un nuevo dataset
    if api_generator is None or not api_generator.ready:
        return redirect(url_for('get_load_0'))

    # Acciones a realizar si se sube el formulario JSON
    if request.form['form'] == "JSON":
        jsonInput = request.form['jsonInput']
        try:
            resultPredictHeaders, resultPredictValues, typeResultHeaders, typeResultValues = api_generator.predictNewValues(
                jsonInput)
        except Exception as e:
            print(e)
            return render_template("predict.html", api=api_generator, error=True)

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
                resultPredictHeaders, resultPredictValues, typeResultHeaders, typeResultValues = api_generator.predictNewValues(
                    file_path, typeData="CSV", separator=request.form['separator'])
            except Exception as e:
                print(e)
                return render_template("predict.html", api=api_generator, error=True)


    return render_template("predict.html", api=api_generator, headers=resultPredictHeaders, dataset=resultPredictValues, typeHeader=typeResultHeaders, typeDataset=typeResultValues)


# Ruta GRAPHS.
@flask_app.route("/graphs")
def graphs():
    global api_generator

    # Si api_generator (la api) no esta definido ni listo, se redirige a la carga de un nuevo dataset
    if api_generator is None or not api_generator.ready:
        return redirect(url_for('get_load_0'))

    # Generate the figure **without using pyplot**.
    figures = api_generator.graphs()
    data = []
    for fig in figures:
        # Save it to a temporary buffer.
        buf = BytesIO()
        fig.savefig(buf, format="png")
        # Embed the result in the html output.
        data.append(base64.b64encode(buf.getbuffer()).decode("ascii"))
    return render_template("graphs.html", api=api_generator, data=data)


"""

    Endpoints de la API REST.

    Todas estas rutas tienen el prefijo '/api/'

"""


@flask_app.route("/api/")
def defaultApiRoute():
    return jsonify({
        "status": "Api is working"
    })


@flask_app.route("/api/dataset")
def datasetApi():
    global api_generator

    if api_generator is None or not api_generator.ready:
        return jsonify({"error": "API is not generated yet"})

    return jsonify(api_generator.filterDataset(request.args))


@flask_app.route("/api/metrics")
def metricsApi():
    global api_generator

    if api_generator is None or not api_generator.ready:
        return jsonify({"error": "API is not generated yet"})

    return jsonify(api_generator.metrics)


@flask_app.route("/api/model")
def modelApi():
    global api_generator

    if api_generator is None or not api_generator.ready:
        return jsonify({"error": "API is not generated yet"})

    return jsonify(api_generator.getModelParams())


@flask_app.route("/api/predict", methods=["POST"])
def predictApi():
    global api_generator

    if api_generator is None or not api_generator.ready:
        return jsonify({"error": "API is not generated yet"})

    return jsonify(api_generator.predictNewValues(request.get_json(), toApi=True))


@flask_app.route("/api/load", methods=["POST"])
def loadApi():
    global api_generator

    data = request.get_json()

    # dataset, inputLabel y modelType son atributos que DEBEN incluir el cuerpo de la petición

    if "dataset" not in data:
        return jsonify({"error": "There is no dataset in the POST body and it must be supplied"})
    if "inputLabel" not in data:
        return jsonify({"error": "There is no inputLabel in the POST body and it must be supplied"})
    if "modelType" not in data:
        return jsonify({"error": "There is no modelType in the POST body and it must be supplied"})

    # Obtiene el dataset
    api_generator = load_json(data["dataset"])

    # Procesa los NaN y los nulos
    if "nanNullMode" in data:
        if data["nanNullMode"] == "fill":
            if "fillvalue" in data:
                api_generator.processNanNull(
                    data["nanNullMode"], data["fillvalue"])
            else:
                return jsonify({"error": "NanNullMode is fill but no fillvalue is supplied"})
        else:
            api_generator.processNanNull(data["nanNullMode"])
    else:
        api_generator.processNanNull("drop")

    # Obtiene el inputLabel
    api_generator.setInputLabel(data["inputLabel"])

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
    api_generator.setAlgorithm(mltype, modelType)

    # Carga los parámetros en la API si estan definidos en el cuerpo de la petición POST
    if "modelParams" in data:
        # Se obtienen los parámetros escogidos en el formulario
        api_generator.setAlgorithmParams(data["modelParams"])
    else:
        api_generator.setAlgorithmParams({})

    # Obtiene las columnas a no tener en cuenta y las carga en la API si estan definidos en el cuerpo de la petición POST
    if "dropColumns" in data:
        api_generator.setDropColumns(data["dropColumns"])
    else:
        api_generator.setDropColumns([])

     # Obtiene el tamaño de test y lo carga en la API si estan definidos en el cuerpo de la petición POST
    if "testSize" in data:
        api_generator.setTestSize(data["testSize"])
    else:
        api_generator.setTestSize(0.3)

    # Si se trata de un problema de clasificación binario, obtiene el valor de la variable objetivo positivo
    # (para los True Positive, False Positive, True Negative y False Negative) y lo carga en la API
    if api_generator.isBinaryClassification:
        if "positiveLabel" in data:
            api_generator.setPositiveLabel(data["positiveLabel"])
        else:
            return jsonify({"error": "positiveLabel must be suplied to Binary Classification problems"})

    api_generator.trainModel()
    api_generator.evaluateModel()

    if "email" in data:
        # Parámetros del correo
        msg = Message(
            'API generation complete', sender='tfgadrianruizparra@gmail.com', recipients=[data["email"]])
        msg.body = "The API has been generated successfully and its currently operable."
        # Envia el correo
        mail.send(msg)

    # El atributo ready de la API se marca a True.
    api_generator.ready = True

    return jsonify({"success": "The API has been successfully generated and its now operable."})
