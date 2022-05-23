import pandas as pd
import numpy as np
import json
import os

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, rand_score, v_measure_score

from sklearn.preprocessing import OneHotEncoder

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift
from sklearn.cluster import Birch


import scikitplot as skplt

def load_csv(csvFile, separator=','):
    """Inicializa un objeto API_Generator a partir de un fichero CSV.

    Args:
        csvFile (str): Path del fichero csv.
        separator (str, optional): Separador de valores del fichero csv. Por defecto a ','.

    Returns:
        API_Generator: Generador de la API.
    """
    datasetDF = pd.read_csv(csvFile, sep=separator)
    return API_Generator(datasetDF)

def load_json(inputJson):
    """Inicializa un objeto API_Generator a partir de un objeto dict JSON.

    Args:
        inputJson (dict): Objeto JSON de tipo dict.

    Returns:
        API_Generator: Generador de la API.
    """
    datasetDF = pd.DataFrame(inputJson)
    return API_Generator(datasetDF)
    


class API_Generator:
    def __init__(self, dataset):
        """Constructor de API_Generator

        Args:
            dataset (Pandas Dataframe): Dataset del experimento del cual se desea generar la API
        """
        self.datasetDF = dataset
        self.step = 1
        self.ready = False

    def processNanNull(self, mode="drop", fillValue=-1):
        """Procesa las filas que contengan valores nulos o valores nan. Tiene dos modos:
        Modo drop: se deshace de las filas
        Modo fill: rellena con fillValue los valores nulos

        Args:
            mode (str, optional): Modo de proceso de las filas. Por defecto a "drop".
            fillValue (int, optional): Si el modo es "fill", es el valor con el que rellena los nulos/nan. Por defecto a -1.
        """
        self.nanNullMode = mode
        if mode=="drop":
            self.datasetDF = self.datasetDF.dropna().reset_index(drop=True)
        elif mode=="fill":
            self.fillValue = fillValue
            self.datasetDF = self.datasetDF.fillna(int(fillValue)).reset_index(drop=True)
            self.__cleanStringColumns()

    def __cleanStringColumns(self):
        """Método privado. Este método limpia las columnas que tras haber rellenado los valores nulos queden
        con tipos mezclados (por ejemplo, columnas con str,int). Si no se limpian, al aplicar One Hot Encoding
        saltará una excepción
        """
        stringColumns = self.datasetDF.select_dtypes(include=[object, 'category'])
        self.datasetDF[stringColumns.columns] = stringColumns.astype(str)

    def setInputLabel(self, inputLabel):
        """Introduce la variable objetivo del experimento.

        Args:
            inputLabel (str): nombre de la columna que contiene la variable objetivo
        """
        self.inputLabel = inputLabel

        if self.nanNullMode != "drop":
            self.__dropMissingInputLabelRows()

    def __dropMissingInputLabelRows(self):
        """Método privado. Este método se deshace de las filas
        cuyo variable objetivo es nulo ya que no aportan información.
        
        Este método se lanzará normalmente cuando el modo Nan Null no sea drop.
        """
        typeInputLabel = self.datasetDF[self.inputLabel].dtype
        if typeInputLabel in ['object', 'category']:
            castedFillValue = str(self.fillValue)
        else:
            castedFillValue = np.cast[typeInputLabel](self.fillValue)
        self.datasetDF = self.datasetDF[self.datasetDF[self.inputLabel] != castedFillValue].reset_index(drop=True)

    def setAlgorithm(self, mltype, modelType):
        """Introduce el tipo de problema ML y el algoritmo del modelo para el experimento

        Args:
            mltype ( str ): Tipo del problema ML.
            modelType ( str ): Algoritmo para el modelo.

        """
        self.mltype = mltype
        self.modelType = modelType
        self.isBinaryClassification = (mltype == "Classification") and (len(self.getPossibleLabels()) == 2)

        if self.mltype == "Classification":
            if self.modelType == "GNB":
                self.model = GaussianNB()
            elif self.modelType == "SVC":
                self.model = SVC()
            elif self.modelType == "KNN":
                self.model = KNeighborsClassifier()
            elif self.modelType == "DT":
                self.model = DecisionTreeClassifier()
            elif self.modelType == "RF":
                self.model = RandomForestClassifier()
            else:
                raise Exception(
                    self.modelType + " no corresponde con ningun algoritmo de clasificacion.")

        elif self.mltype == "Regression":
            if self.modelType == "LR":
                self.model = LinearRegression()
            elif self.modelType == "SVR":
                self.model = SVR()
            elif self.modelType == "SGDR":
                self.model = SGDRegressor()
            elif self.modelType == "KR":
                self.model = KernelRidge()
            elif self.modelType == "GBR":
                self.model = GradientBoostingRegressor()
            else:
                raise Exception(
                    self.modelType + " no corresponde con ningun algoritmo de regresion.")

        elif self.mltype == "Clustering":
            if self.modelType == "KM":
                self.model = KMeans()
            elif self.modelType == "AP":
                self.model = AffinityPropagation()
            elif self.modelType == "MS":
                self.model = MeanShift()
            elif self.modelType == "B":
                self.model = Birch()
            else:
                raise Exception(
                    self.modelType + " no corresponde con ningun algoritmo de clustering.")
        else:
            raise Exception("No se reconoce el problema " + self.mltype + ".")

    def getAlgorithmParams(self):
        """Retorna los parámetros del algoritmo del modelo

        Returns:
            dict: Parámetros del algoritmo del modelo
        """
        return self.model.get_params()

    def setAlgorithmParams(self, algorithmParams):
        """Introduce nuevos parámetros para el algoritmo del modelo

        Args:
            algorithmParams (dict): Nuevos parámetros del algoritmo
        """
        self.algorithmParams = algorithmParams
        self.model.set_params(**algorithmParams)

    def setDropColumns(self,  dropColumns=[]):
        """Introduce las columnas a no tener en cuenta del experimento.
        Ademas, las columnas restantes serán asignadas como atributos (features)

        Args:
            dropColumns (list, optional): Lista de columnas a no tener en cuenta. Por defecto a [].
        """
        self.dropColumns = dropColumns
        self.datasetDF = self.datasetDF.drop(dropColumns, axis=1)

        self.features = self.datasetDF.drop(self.inputLabel, axis=1).columns

    def setTestSize(self, testSize=0.3):
        """Introduce el tamaño del test del experimento

        Args:
            testSize (float, optional): Tamaño del test. Por defecto a 0.3.
        """
        self.testSize = float(testSize)

    def setPositiveLabel(self, positiveLabel=1):
        """Introduce el valor de la variable objetivo positivo para los experimentos de clasificación binaria.
           Gracias a este valor positivo, se podrán realizar los cálculos de TP,FP,TN,FN

        Args:
            positiveLabel (any, optional): Valor de la variable objetivo positivo. Por defecto a 1.
        """
        typeLabel = type(self.getPossibleLabels()[0])
        self.positiveLabel = typeLabel(positiveLabel)

    def trainModel(self):
        """ Entrena el modelo con los parámetros de la instancia del objeto.
        """

        self.enc = OneHotEncoder(handle_unknown='ignore')

        self.x = self.__ohe_encode(self.datasetDF.drop(
            self.inputLabel, axis=1), isTraining=True)
        self.y = self.datasetDF[self.inputLabel]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y, test_size=self.testSize, random_state=42)

        self.model.fit(self.x_train, self.y_train)

    def evaluateModel(self):
        """Evalúa el modelo con los parámetros de la instancia del objeto generando así las métricas.
        """

        self.predictions = self.model.predict(self.x_test)
        self.metrics = {}

        if self.mltype == "Classification":
            self.metrics["accuracy"] = accuracy_score(self.y_test, self.predictions)
            if self.isBinaryClassification:  # Binary Classification
                self.metrics["precision"] = precision_score(
                    self.y_test, self.predictions, pos_label=self.positiveLabel)
                self.metrics["recall"] = recall_score(
                    self.y_test, self.predictions, pos_label=self.positiveLabel)
                self.metrics["f1"] = f1_score(
                    self.y_test, self.predictions, pos_label=self.positiveLabel)
            else:  # Multi-Label Classification
                self.metrics["precision"] = precision_score(
                    self.y_test, self.predictions, average="micro")
                self.metrics["recall"] = recall_score(
                    self.y_test, self.predictions, average="macro")
                self.metrics["f1"] = f1_score(
                    self.y_test, self.predictions, average="macro")
            self.metrics["confusion_matrix"] = confusion_matrix(
                self.y_test, self.predictions).tolist()

        elif self.mltype == "Regression":
            self.metrics["MAE"] = mean_absolute_error(self.y_test, self.predictions)
            self.metrics["MSE"] = mean_squared_error(self.y_test, self.predictions)
            self.metrics["RMSE"] = np.sqrt(
                mean_squared_error(self.y_test, self.predictions))
            self.metrics["RMSLE"] = np.log(
                np.sqrt(mean_squared_error(self.y_test, self.predictions)))
            self.metrics["R2"] = r2_score(self.y_test, self.predictions)

        elif self.mltype == "Clustering":
            self.metrics["silhouette_coefficient"] = silhouette_score(
                self.x_train, self.model.labels_)
            self.metrics["calinski_harabaz"] = calinski_harabasz_score(
                self.x_train, self.model.labels_)
            self.metrics["davies_bouldin"] = davies_bouldin_score(
                self.x_train, self.model.labels_)
            self.metrics["rand_score"] = rand_score(self.y_test, self.predictions)
            self.metrics["v-measure"] = v_measure_score(
                self.y_test, self.predictions)

    def getAlgorithm(self):
        algorithms = {
            "GNB": "GaussianNB",
            "SVC": "SVC",
            "KNN": "KNeighborsClassifier",
            "DT": "DecisionTreeClassifier",
            "RF": "RandomForestClassifier",

            "LR": "LinearRegression",
            "SVR": "SVR",
            "SGDR": "SGDRegressor",
            "KR": "KernelRidge",
            "GBR": "GradientBoostingRegressor",

            "KM": "KMeans",
            "AP": "AffinityPropagation",
            "MS": "MeanShift",
            "B": "Birch"
        }

        return algorithms[self.modelType]


    def getModelParams(self):
        """Devuelve los parámetros del modelo del experimento

        Returns:
            dict: Parámetros del modelo del experimento
        """


        modelParams = {}
        modelParams["label"] = self.inputLabel
        modelParams["features"] = self.features.tolist()
        modelParams["problem"] = self.mltype

        if self.mltype == "Classification":
            if self.isBinaryClassification:
                modelParams["classification"] = "Binary"

                try:
                    modelParams["possitive_label"] = float(self.positiveLabel)
                except ValueError:
                    try:
                        modelParams["possitive_label"] = int(
                            self.positiveLabel)
                    except ValueError:
                        modelParams["possitive_label"] = str(
                            self.positiveLabel)

            else:
                modelParams["classification"] = "Multi-Label"

        if self.mltype == "Classification" or self.mltype == "Clustering":
            modelParams["labels"] = self.getPossibleLabels().tolist()

        modelParams["NanNull"] = self.nanNullMode
        if self.nanNullMode == "fill":
            modelParams["fillValue"] = self.fillValue

        modelParams["dropped"] = self.dropColumns
        modelParams["algorithm"] = self.getAlgorithm()
        modelParams["algorithm_args"] = self.algorithmParams


        modelParams["dataset_size"] = self.datasetDF.shape[0]
        modelParams["training_size"] = self.x_train.shape[0]
        modelParams["testing_size"] = self.x_test.shape[0]
        return modelParams


    def getColumns(self):
        """Devuelve las columnas del dataset del experimento

        Returns:
            list: Columnas del dataset
        """
        return self.datasetDF.columns

    def getFeatures(self):
        """Devuelve las columnas (excepto la variable objetivo) del dataset del experimento
        Returns:
            list: Columnas no objetivo del dataset
        """
        return self.datasetDF.drop(self.inputLabel, axis=1)

    def getPossibleLabels(self):
        """Devuelve los posibles valores de la variable objetivo. Utilizado para experimentos de Clasificación.

        Returns:
            numpy.ndarray: Posibles valores de la variable objetivo.
        """
        return self.datasetDF[self.inputLabel].unique()

    def getValues(self):
        """Devuelve los valores de las filas del dataset del experimento

        Returns:
            numpy.ndarray: Valores de las filas del dataset
        """
        return self.datasetDF.values

    def getProblem(self):
        """Devuelve el problema ML del experimento

        Returns:
            str: Problema ML
        """
        if self.mltype == "Classification":
            if self.isBinaryClassification:
                return "Binary Classification"
            else:
                return "Multi-Label Classification"
        else:
            return self.mltype

    def predictNewValues(self, inputData, typeData="JSON", separator=",", toApi = False):
        """Predice nuevos valores utilizando el modelo del experimento

        Args:
            inputData (str, dict): Si typeData == "JSON", contiene un str o un dict con los valores del JSON. Si typeData =="CSV", contiene el path del fichero CSV
            typeData (str, optional): Tipo de entrada del parámetro inputData. Por defecto a "JSON".
            separator (str, optional): Si typeData == "CSV", indica el separador entre valores del fichero CSV. Por defecto a ",".
            toApi (bool, optional): Indica si la predicción proviene del endpoint de la API Rest. Por defecto a False.

        Returns:
            list: Lista de valores predichos
            Pandas Dataframe: Dataframe de entrada con la columna result indicando los valores predichos
            Pandas Dataframe: Dataframe auxiliar que indica que columnas fueron usadas y descartadas, si está la columna objetivo y las filas que contienen nulos
        """

        if typeData == "JSON":
            if isinstance(inputData, str):
                inputData = json.loads(inputData)
            inputDf = self.__dataframe_from_input_json(inputData)

        elif typeData == "CSV":
            inputDf = pd.read_csv(inputData, sep=separator)

       
        predictInputDf = inputDf.drop(self.inputLabel, axis=1, errors='ignore')
        predictInputDf = predictInputDf.drop(self.dropColumns, axis=1, errors='ignore')

         # Ordena las columnas para que coincidan con el entrenamiento
        predictInputDf = predictInputDf.reindex(self.features, axis=1)

        cleanPredictInputDf, nullRows = self.__clean_nan_null_input_dataframe(predictInputDf)

        result = self.__predict_input_dataframe(cleanPredictInputDf).tolist()

        # Para la API, se devuelve la lista de valores predichos
        if toApi:
            return result

        # Para la app web, se devuelve el dataframe con una nueva columna de valores predichos.

        
        if self.nanNullMode == "drop" and any(nullRows):
            type(nullRows[0])
            resultIndex = 0
            for i in range(len(nullRows)):
                if nullRows[i]:
                    inputDf.at[i, "result"] = "DROPPED"
                else:
                    inputDf.at[i, "result"] = result[resultIndex]
                    resultIndex+=1
        
        else:
            inputDf["result"] = result

        typeDataframe = inputDf.copy()


        typeDataframe[self.features] = "Feature"

        for column in self.dropColumns:
            if column in typeDataframe.columns:
                typeDataframe[column] = "NotUsed"
                typeDataframe = typeDataframe.rename(columns = {column: "NotUsed"})

        if self.inputLabel in typeDataframe.columns:
            typeDataframe[self.inputLabel] = "Label"
            typeDataframe = typeDataframe.rename(columns = {self.inputLabel: "Label"})
        
        typeDataframe[inputDf[self.features].isna().any(axis=1)] = "NullRow"

        return inputDf.columns, inputDf.values, typeDataframe.columns, typeDataframe.values

    def filterDataset(self, args):
        """Retorna el dataset como JSON del experimento aplicando los filtros pasados por parámetros

        Args:
            args (dict): Parámetros a filtrar

        Returns:
            (dict): JSON con el dataset del experimento filtrado
        """
        # Cargar en returnDf una copia del dataset completo
        returnDf = self.datasetDF.copy()

        # Filtrar returnDf con los parametros de la query
        for key, value in args.items():
            try:
                valor = float(value)
            except ValueError:
                try:
                    valor = int(value)
                except ValueError:
                    valor = str(value)

            returnDf = returnDf.loc[returnDf[key] == valor]

        return json.loads(returnDf.to_json(orient="records"))


    def downloadCSV(self, path):
        """Exporta el dataset a CSV retornando el nombre del fichero

        Args:
            path (str): Path del nuevo fichero CSV que se desea exportar

        Returns:
            str: Nombre del fichero exportado
        """
        filename = self.inputLabel + '_' + self.mltype + '_' + self.modelType + '.csv'
        filepath = os.path.join(path,filename)
        self.datasetDF.to_csv(filepath, index = False)
        return filename

    def graphs(self):
        """Genera gráficos del experimento en función de los parámetros
        y los retorna como figuras gráficas (plots).

        Returns:
            list[Matplotlib Pyplot Figure]: Lista de figuras que contienen los gráficos a renderizar
        """
        returnGraphs = []
        if self.mltype == "Classification":         
            returnGraphs.append(skplt.metrics.plot_confusion_matrix(self.y_test, self.predictions).get_figure())
            if not hasattr(self.model, 'probability') or self.model.probability:
                returnGraphs.append(skplt.metrics.plot_roc(self.y_test, self.model.predict_proba(self.x_test)).get_figure())
        elif self.mltype == "Clustering":
            cluster_labels = self.model.predict(self.x)
            returnGraphs.append(skplt.metrics.plot_silhouette(self.x, cluster_labels).get_figure())
            if hasattr(self.model, 'n_clusters') and hasattr(self.model, 'score'):
                returnGraphs.append(skplt.cluster.plot_elbow_curve(self.model,self.x,cluster_ranges=range(1, 10)).get_figure())

        if self.mltype == "Classification" or self.mltype == "Regression":
            returnGraphs.append(skplt.estimators.plot_learning_curve(self.model, self.x, self.y).get_figure())


        return returnGraphs

    def __dataframe_from_input_json(self, inputJson):
        """Método privado que construye un DataFrame a partir de un JSON

        Args:
            inputJson (dict, list[dict]): Objeto JSON

        Returns:
            Pandas Dataframe: Dataframe construido a partir del objeto JSON
        """
        # Si en lugar de una lista de objetos JSON se pasa un unico JSON (dict), es necesario
        # transformarlo a una lista de JSON
        if isinstance(inputJson, dict):
            inputJson = [inputJson]

        # DataFrame de los nuevos datos
        df = pd.DataFrame(inputJson)

        return df

    def __clean_nan_null_input_dataframe(self, inputDf):
        """Metodo privado. Este método se encarga de procesar los
        datos nan y nulos recibidos para predecir nuevos valores.

        Args:
            inputDf (Pandas Dataframe): Datos de entrada para ser procesados y limpiados.

        Returns:
            Pandas Dataframe: Datos de entradas limpios y procesados.
            List[boolean]: Si el modo de trato de nan/nulos es "drop", representa las filas que han sido deshechadas (y por tanto,
            no cuentan con ninguna predicción).
        """
        
        nullRowsDf = inputDf.isnull().any(axis=1)

        nullRows = nullRowsDf.tolist()

        if self.nanNullMode == "drop":
            inputDf = inputDf.dropna().reset_index(drop=True)
        elif self.nanNullMode=="fill":
            inputDf = inputDf.fillna(int(self.fillValue)).reset_index(drop=True)
            stringColumns = inputDf.select_dtypes(include=[object, 'category'])
            inputDf[stringColumns.columns] = stringColumns.astype(str)

        return inputDf, nullRows

    def __predict_input_dataframe(self, inputDf):
        """Método privado que predice nuevos valores a partir de un Dataframe de entrada

        Args:
            inputDf (Dataframe Pandas): Dataframe de entrada con las filas para predecir

        Returns:
            list: Valores predichos a partir de las filas del Dataframe de entrada
        """

        predictDF = self.__ohe_encode(inputDf)

        # Se realiza la predicción utilizando el modelo entrenado
        resultado = self.model.predict(predictDF)

        return resultado

    def __ohe_encode(self, x, isTraining=False):
        """Método privado que entrena o transforma las filas categóricas del dataset del experimento o del dataframe a predecir. Utiliza One-Hot-Encoding

        Args:
            x (Pandas Dataframe): Dataset del experimento o Dataframe a predecir.
            isTraining (bool, optional): Indica si el método se ejecuta en fase de entrenamiento o durante una predicción. Por defecto a False.

        Returns:
            Pandas Dataframe: Dataframe pasado por parámetro de entrada al cual se le ha aplicado One Hot Encoding.
        """
        # Obtiene las columnas discretas/categóricas
        categoricalFeatures = x.select_dtypes(include=[object, 'category'])

        # Obtiene un Dataframe con One-Hot-Encoding aplicado

        if isTraining:
            oneHotDF = pd.DataFrame(self.enc.fit_transform(categoricalFeatures).toarray(
            ), columns=self.enc.get_feature_names(categoricalFeatures.columns))

        else:
            oneHotDF = pd.DataFrame(self.enc.transform(categoricalFeatures).toarray(
            ), columns=self.enc.get_feature_names(categoricalFeatures.columns))

        # Se deshace de las columnas originales y se unen las columnas aplicadas con One-Hot-Encoding.
        return x.drop(categoricalFeatures, axis=1).join(oneHotDF)
