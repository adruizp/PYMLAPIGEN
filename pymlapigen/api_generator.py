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
    """Instances an API_Generator object from a CSV file.

    Args:
        csvFile (str): CSV file's path
        separator (str, optional): Values separator in the CSV fil. Defaults to ','.

    Returns:
        API_Generator: API Generator.
    """
    datasetDF = pd.read_csv(csvFile, sep=separator)
    return API_Generator(datasetDF)


def load_json(inputJson):
    """Instances an API_Generator object from a JSON dict.

    Args:
        inputJson (dict): JSON dict.

    Returns:
        API_Generator: API Generator.
    """
    datasetDF = pd.DataFrame(inputJson)
    return API_Generator(datasetDF)


class API_Generator:
    def __init__(self, dataset):
        """API_Generator's constructor

        Args:
            dataset (Pandas Dataframe): Dataset of the experiment.
        """
        self.datasetDF = dataset
        self.step = 1
        self.ready = False

    def processNanNull(self, mode="drop", fillValue=-1):
        """Processes the missing values from the dataset. There are three modes:
        Drop mode: deletes rows with missing values.
        Fill mode: fills missing values with a value.
        Mean mode: fills missing values with the mean of the column.

        Args:
            mode (str, optional): Processing missing values mode. Defaults to "drop".
            fillValue (int, optional): If "fill" mode, represents the filling value. Defaults to -1.
        """
        self.nanNullMode = mode
        if mode == "drop":
            self.datasetDF = self.datasetDF.dropna().reset_index(drop=True)
        elif mode == "fill":
            self.fillValue = fillValue
            self.datasetDF = self.datasetDF.fillna(
                int(fillValue)).reset_index(drop=True)
            self.__cleanStringColumns()
        elif mode == "mean":
            self.__replaceMean(self.datasetDF)

    def __cleanStringColumns(self):
        """Private method. This method cleans the columns after missing values are filled in case they are mixed types
        (for example, columns with str,int values). If not cleaned, an exception will occur in One Hot Encoding.
        """
        stringColumns = self.datasetDF.select_dtypes(
            include=[object, 'category'])
        self.datasetDF[stringColumns.columns] = stringColumns.astype(str)

    def __replaceMean(self, dataset):
        """Private method. This method fills the missing values with the column mean (numeric values) or
        the most common word (string values)
        Args:
            dataset (Pandas Dataframe): Dataset to fill.
        """
        for column in dataset:
            if len(dataset[column].dropna()) == 0:
                # If the column has no values (all null), it will be dropped.
                dataset.drop(column, axis=1)
            else:
                if dataset[column].dtype in ['object', 'category']:
                    # Most common value
                    dataset[column] = dataset[column].fillna(
                        dataset[column].mode()[0])
                else:
                    # Mean value
                    dataset[column] = dataset[column].fillna(
                        dataset[column].mean())

    def setInputLabel(self, inputLabel):
        """Inputs the label of the experiment.

        Args:
            inputLabel (str): Name of the label column
        """
        self.inputLabel = inputLabel

        if self.nanNullMode == "fill":
            self.__dropMissingInputLabelRows()

    def __dropMissingInputLabelRows(self):
        """Private method. This method gets rid of the rows
        whose input label is null.
        """
        typeInputLabel = self.datasetDF[self.inputLabel].dtype
        if typeInputLabel in ['object', 'category']:
            castedFillValue = str(self.fillValue)
        else:
            castedFillValue = np.cast[typeInputLabel](self.fillValue)
        self.datasetDF = self.datasetDF[self.datasetDF[self.inputLabel]
                                        != castedFillValue].reset_index(drop=True)

    def setAlgorithm(self, mltype, modelType):
        """Sets the Machine Learning problem and the model's algorithm

        Args:
            mltype ( str ): Machine Learning Problem.
            modelType ( str ): Model's Algorithm.

        """
        self.mltype = mltype
        self.modelType = modelType
        self.isBinaryClassification = (mltype == "Classification") and (
            len(self.getPossibleLabels()) == 2)

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
                    self.modelType + " is not related with any classification algorithm.")

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
                    self.modelType + " is not related with any regresion algorithm.")

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
                    self.modelType + " is not related with any clustering algorithm.")
        else:
            raise Exception(
                "Unknown Machine Learning problem " + self.mltype + ".")

    def getAlgorithmParams(self):
        """Gets the model algorithm parameters

        Returns:
            dict: Algorithm parameters
        """
        return self.model.get_params()

    def setAlgorithmParams(self, algorithmParams):
        """Sets the model algorithm parameters

        Args:
            algorithmParams (dict): Algorithm parameters
        """
        self.algorithmParams = algorithmParams
        self.model.set_params(**algorithmParams)

    def setDropColumns(self,  dropColumns=[]):
        """Sets the dropping columns (columns that will be deleted from the dataset)
        The rest of columns are designed as features.

        Args:
            dropColumns (list, optional): Dropping columns list. Defaults to [].
        """
        self.dropColumns = dropColumns
        self.datasetDF = self.datasetDF.drop(dropColumns, axis=1)

        if self.mltype != "Clustering":
            self.features = self.datasetDF.drop(self.inputLabel, axis=1).columns
        else:
            self.features = self.datasetDF.columns

    def setTestSize(self, testSize=0.3):
        """Sets the test set size percentage

        Args:
            testSize (float, optional): Set test size percentage. Defaults to 0.3.
        """
        self.testSize = float(testSize)

    def setPositiveLabel(self, positiveLabel=1):
        """Introduce el valor de la variable objetivo positivo para los experimentos de clasificación binaria.
           Gracias a este valor positivo, se podrán realizar los cálculos de TP,FP,TN,FN

        Args:
            positiveLabel (any, optional): Valor de la variable objetivo positivo. Defaults to 1.
        """
        typeLabel = type(self.getPossibleLabels()[0])
        self.positiveLabel = typeLabel(positiveLabel)

    def trainModel(self):
        """ Trains the API's model.
        """

        self.enc = OneHotEncoder(handle_unknown='ignore')

        if self.mltype != "Clustering":
            self.x = self.__ohe_encode(self.datasetDF.drop(
                self.inputLabel, axis=1), isTraining=True)
            self.y = self.datasetDF[self.inputLabel]
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                self.x, self.y, test_size=self.testSize, random_state=42)

            self.model.fit(self.x_train, self.y_train)
        
        else:
            self.x = self.__ohe_encode(self.datasetDF, isTraining=True)
            self.x_train, self.x_test= train_test_split(
                self.x, test_size=self.testSize, random_state=42)

            self.model.fit(self.x_train)

    def evaluateModel(self):
        """ Evaluates the API's model generating metrics.
        """

        self.predictions = self.model.predict(self.x_test)
        self.metrics = {}

        if self.mltype == "Classification":
            self.metrics["accuracy"] = accuracy_score(
                self.y_test, self.predictions)
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
            self.metrics["MAE"] = mean_absolute_error(
                self.y_test, self.predictions)
            self.metrics["MSE"] = mean_squared_error(
                self.y_test, self.predictions)
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
            #self.metrics["rand_score"] = rand_score(
            #    self.y_test, self.predictions)
            #self.metrics["v-measure"] = v_measure_score(
            #    self.y_test, self.predictions)

    def getAlgorithm(self):
        """Returns the model's algorithm full name

        Returns:
            str: Model's algorithm name
        """
        algorithms = {
            "GNB": "GaussianNB",
            "SVC": "Support Vector Classifier",
            "KNN": "KNeighborsClassifier",
            "DT": "DecisionTreeClassifier",
            "RF": "RandomForestClassifier",

            "LR": "LinearRegression",
            "SVR": "Support Vector Regression",
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
        """Returns the API's model parameters

        Returns:
            dict: API's model parameters
        """

        modelParams = {}
        if self.mltype != "Clustering":
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

        if self.mltype == "Classification":
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
        """Return the dataset columns name

        Returns:
            list: Dataset columns name
        """
        return self.datasetDF.columns

    def getFeatures(self):
        """Returns the dataset feature columns
        Returns:
            list: Dataset feature columns
        """
        if self.mltype != "Clustering":
            return self.datasetDF.drop(self.inputLabel, axis=1)
        else:
            return self.datasetDF

    def getInputLabel(self):
        """Returns the input label name

        Returns:
            str: Input label name
        """
        if self.mltype != "Clustering":
            return self.inputLabel
        else:
            return ""

    def getPossibleLabels(self):
        """Returns the unique values of the label column. Used in Classification problems.

        Returns:
            numpy.ndarray: Unique values of the label column.
        """
        if self.mltype == "Classification":
            return self.datasetDF[self.inputLabel].unique()
        else:
            return None

    def getValues(self):
        """Returns the rows of the dataset

        Returns:
            numpy.ndarray: Dataset value rows
        """
        return self.datasetDF.values

    def getProblem(self):
        """Returns the Machine Learning problem of the API

        Returns:
            str: Machine Learning Problem
        """
        if self.mltype == "Classification":
            if self.isBinaryClassification:
                return "Binary Classification"
            else:
                return "Multi-Label Classification"
        else:
            return self.mltype

    def getPredictions(self):
        return self.predictions

    def getTestSet(self):
        """Returns the API's test set, the label values and predicted values

        Returns:
            noOhe (Pandas Dataframe): Test set feature values.
            y_test (Pandas Series): Test set label values.
            predictions (List): Test set label predicted values.
        """

        if self.mltype != "Clustering":
            noOHE = self.datasetDF.iloc[self.x_test.index].drop(
                self.inputLabel, axis=1)
            return noOHE, self.y_test, self.predictions

        else:
            noOHE = self.datasetDF.iloc[self.x_test.index]
            return noOHE, self.predictions
        

       

    
        

    def predictNewValues(self, inputData, typeData="JSON", separator=",", toApi=False):
        """Predicts new data using the trained model

        Args:
            inputData (str, dict): If typeData == "Input", it consists of a dict with the input data. If typeData == "JSON", it consists of an str or dict with JSON values. If typeData =="CSV", it contains the CSV file path
            typeData (str, optional): Input data type. Defaults to "JSON".
            separator (str, optional): If typeData == "CSV", it represents the CSV file data separator. Defaults to ",".
            toApi (bool, optional): Tells whether the predicción comes from the API JSON endpoint or not. Defaults to False.

        Returns:
            list: List of predicted values
            Pandas Dataframe: Input Dataframe with a result column appended
            Pandas Dataframe: Auxiliar Dataframe that points the type of the result cells
        """

        print(typeData)

        if typeData == "Input":
            
            #Clean empty data
            inputData = {k: v for k, v in inputData.items() if v}

            for key in inputData:
                if inputData[key] == "":
                    inputData.pop(key)
                else:
                    typeLabel = self.datasetDF[key].dtype
                    if typeLabel not in ['object', 'category']:
                        inputData[key] = np.cast[typeLabel](inputData[key]).item()
            
            inputDf = self.__dataframe_from_input_json(inputData)
    

        elif typeData == "JSON":
            if isinstance(inputData, str):
                inputData = json.loads(inputData)
            inputDf = self.__dataframe_from_input_json(inputData)

        elif typeData == "CSV":
            inputDf = pd.read_csv(inputData, sep=separator)

        predictInputDf = inputDf

        if self.mltype != "Clustering":
            predictInputDf = predictInputDf.drop(self.inputLabel, axis=1, errors='ignore')
        predictInputDf = predictInputDf.drop(
            self.dropColumns, axis=1, errors='ignore')

        # Orders the columns so it matches the train one
        predictInputDf = predictInputDf.reindex(self.features, axis=1)

        cleanPredictInputDf, nullRows = self.__clean_nan_null_input_dataframe(
            predictInputDf)

        print(cleanPredictInputDf)
        print(cleanPredictInputDf.shape)
        print(cleanPredictInputDf.shape[0])

        if cleanPredictInputDf.shape[0] > 0:
            result = self.__predict_input_dataframe(cleanPredictInputDf).tolist()

        #  Returns the result if the petition was from the API JSON endpoint
        if toApi:
            return result

        # Returns the input dataframe with a result column appended

        if self.nanNullMode == "drop" and any(nullRows):
            type(nullRows[0])
            resultIndex = 0
            for i in range(len(nullRows)):
                if nullRows[i]:
                    inputDf.at[i, "result"] = "DROPPED"
                else:
                    inputDf.at[i, "result"] = result[resultIndex]
                    resultIndex += 1

        else:
            inputDf["result"] = result

        typeDataframe = inputDf.copy()

        typeDataframe[self.features] = "Feature"

        

        for column in self.dropColumns:
            if column in typeDataframe.columns:
                typeDataframe[column] = "NotUsed"
                typeDataframe = typeDataframe.rename(
                    columns={column: "NotUsed"})

        if self.mltype != "Clustering" and self.inputLabel in typeDataframe.columns:
            typeDataframe[self.inputLabel] = "Label"
            typeDataframe = typeDataframe.rename(
                columns={self.inputLabel: "Label"})

        features = list(set(inputDf.columns) & set(self.features))
        typeDataframe[inputDf[features].isna().any(axis=1)] = "NullRow"

        return inputDf.columns, inputDf.values, typeDataframe.columns, typeDataframe.values

    def filterDataset(self, args):
        """Returns the dataset as JSON filtered with args

        Args:
            args (dict): Filter params

        Returns:
            (dict): JSON filtered dataset
        """
        returnDf = self.datasetDF.copy()

        # Filters
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
        """Exports the dataset into a CSV file

        Args:
            path (str): Path of the file to export 

        Returns:
            str: Exported file name
        """
        filename = self.inputLabel + '_' + self.mltype + '_' + self.modelType + '.csv'
        filepath = os.path.join(path, filename)
        self.datasetDF.to_csv(filepath, index=False)
        return filename

    def graphs(self):
        """Generates plots of the experiment and returns it as 
        figures

        Returns:
            list[Matplotlib Pyplot Figure]: Figures plot list
        """
        returnGraphs = []
        if self.mltype == "Classification":
            returnGraphs.append(skplt.metrics.plot_confusion_matrix(
                self.y_test, self.predictions).get_figure())
            if not hasattr(self.model, 'probability') or self.model.probability:
                returnGraphs.append(skplt.metrics.plot_roc(
                    self.y_test, self.model.predict_proba(self.x_test)).get_figure())
        elif self.mltype == "Clustering":
            cluster_labels = self.model.predict(self.x)
            returnGraphs.append(skplt.metrics.plot_silhouette(
                self.x, cluster_labels).get_figure())
            if hasattr(self.model, 'n_clusters') and hasattr(self.model, 'score'):
                returnGraphs.append(skplt.cluster.plot_elbow_curve(
                    self.model, self.x, cluster_ranges=range(1, 10)).get_figure())

        if self.mltype == "Classification" or self.mltype == "Regression":
            returnGraphs.append(skplt.estimators.plot_learning_curve(
                self.model, self.x, self.y).get_figure())

        return returnGraphs

    def __dataframe_from_input_json(self, inputJson):
        """Private method that builds a Dataframe from a JSON dict list

        Args:
            inputJson (dict, list[dict]): JSON object

        Returns:
            Pandas Dataframe: Built Dataframe
        """
        # If input is a single dict, cast it into a list of dicts
        if isinstance(inputJson, dict):
            inputJson = [inputJson]

        # Build Dataframe
        df = pd.DataFrame(inputJson)

        return df

    def __clean_nan_null_input_dataframe(self, inputDf):
        """Metodo privado. Cleans the missing value in order to make predictions.

        Args:
            inputDf (Pandas Dataframe): Input data to be cleaned.

        Returns:
            Pandas Dataframe: Cleaned input data.
            List[boolean]: If nan/null mode is "drop", represents the dropped rows 
        """

        nullRowsDf = inputDf.isnull().any(axis=1)

        nullRows = nullRowsDf.tolist()

        if self.nanNullMode == "drop":
            inputDf = inputDf.dropna().reset_index(drop=True)
        elif self.nanNullMode == "fill":
            inputDf = inputDf.fillna(
                int(self.fillValue)).reset_index(drop=True)
            stringColumns = inputDf.select_dtypes(include=[object, 'category'])
            inputDf[stringColumns.columns] = stringColumns.astype(str)
        elif self.nanNullMode == "mean":
            self.__replaceMean(inputDf)

        return inputDf, nullRows

    def __predict_input_dataframe(self, inputDf):
        """Private method. Predicts new data from an input Dataframe

        Args:
            inputDf (Dataframe Pandas): Input data Dataframe 

        Returns:
            list: Predicted values list
        """

        predictDF = self.__ohe_encode(inputDf)

        # Predicts using the trained model
        resultado = self.model.predict(predictDF)

        return resultado

    def __ohe_encode(self, x, isTraining=False):
        """Private method. Transforms the non numeric data into numeric using One Hot Encoding.
        Args:
            x (Pandas Dataframe): Input features Dataframe
            isTraining (bool, optional): Tells if the transformation belongs to a training or prediction. Defaults to False.

        Returns:
            Pandas Dataframe: Dataframe with One-Hot-Encoding applied.
        """
        # Gets the discrete/categorical columns
        categoricalFeatures = x.select_dtypes(include=[object, 'category'])

        # Applies One-Hot-Encoding and save it into a new Dataframe

        if isTraining:
            oneHotDF = pd.DataFrame(self.enc.fit_transform(categoricalFeatures).toarray(
            ), columns=self.enc.get_feature_names(categoricalFeatures.columns))

        else:
            oneHotDF = pd.DataFrame(self.enc.transform(categoricalFeatures).toarray(
            ), columns=self.enc.get_feature_names(categoricalFeatures.columns))

        # Gets rid of the old categorical columns and appends the new columns with One-Hot-Encoding applied.
        return x.drop(categoricalFeatures, axis=1).join(oneHotDF)
