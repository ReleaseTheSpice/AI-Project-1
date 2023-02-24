###############################################################
### Martin Van Laethem  - Aiden Lonquist    - Cole Atkinson ###
### A01189452           - A01166561         - A00000000     ###
###                 AI Project 1 - NIDS                     ###
###############################################################

import pandas
from sklearn.model_selection import train_test_split
from sklearn import metrics # is used to create classification results
from sklearn import preprocessing # is used to encode the data
import time
from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
import numpy
import sys
import pickle

class NIDS:
    """class for the network intrusion detection system"""

    def __init__(self, dataFile, classificationMethod, targetTask, model=None):
        """Constructor"""
        colNames = ["srcip","sport","dstip","dsport","proto","state","dur","sbytes","dbytes","sttl","dttl","sloss",
                    "dloss","service","Sload","Dload","Spkts","Dpkts","swin","dwin","stcpb","dtcpb","smeansz","dmeansz",
                    "trans_depth","res_bdy_len","Sjit","Djit","Stime","Ltime","Sintpkt","Dintpkt","tcprtt","synack",
                    "ackdat","is_sm_ips_ports","ct_state_ttl","ct_flw_http_mthd","is_ftp_login","ct_ftp_cmd",
                    "ct_srv_src","ct_srv_dst","ct_dst_ltm","ct_src_ltm","ct_src_dport_ltm","ct_dst_sport_ltm",
                    "ct_dst_src_ltm","attack_cat","Label"]

        self.modelName = "cat-"

        data = pandas.read_csv(dataFile, names=colNames, skiprows=1)

        # ct_flw_http_mthd and is_ftp_login have a lot of unfilled cells, remove them
        data = data.drop(['ct_flw_http_mthd', 'is_ftp_login'], axis=1)
        colNames.remove('ct_flw_http_mthd')
        colNames.remove('is_ftp_login')
        featureCols = colNames[:-2]

        # ct_ftp_cmd has a lot of unfilled cells, set them to '0'
        data['ct_ftp_cmd'] = pandas.to_numeric(data['ct_ftp_cmd'], errors='coerce')
        data['ct_ftp_cmd'] = data['ct_ftp_cmd'].fillna(0)

        # sport and dsport have some non-int values, replace those with 0
        data['sport'] = pandas.to_numeric(data['sport'], errors='coerce')
        data['sport'] = data['sport'].fillna(0)
        data['dsport'] = pandas.to_numeric(data['dsport'], errors='coerce')
        data['dsport'] = data['dsport'].fillna(0)

        # Encode the source and destination IPs
        data['srcip'] = preprocessing.LabelEncoder().fit_transform(data['srcip'])
        data['dstip'] = preprocessing.LabelEncoder().fit_transform(data['dstip'])

        # Convert other non-numeric columns to ints
        data['attack_cat'] = pandas.factorize(data['attack_cat'])[0]
        data['proto'] = pandas.factorize(data['proto'])[0]
        data['state'] = pandas.factorize(data['state'])[0]
        data['service'] = pandas.factorize(data['service'])[0]

        if targetTask == 'Label':
            #Do something
            print("Label")
        elif targetTask == 'attack_cat':
            # # Make a new dataframe for predicting the attack category
            # # It has only the rows that have an attack_cat (drop all rows where atkCat is null)
            # atkCatData = data
            # atkCatData.dropna(axis='rows', subset=['attack_cat'], inplace=True)
            # We are trying to predict the attack category, so remove any rows where it is null
            data.dropna(axis='rows', subset=['attack_cat'], inplace=True)
        else:
            print(f"Task not recognized: {targetTask}.  Please choose 'Label' or 'attack_cat'.")
            sys.exit()


        self.recursiveFeatureElimination(data, targetTask, featureCols, 10)

        #self.linearRegressionAnalysis(data, targetTask, featureCols)

        #self.principalComponentAnalysis(data, targetTask, featureCols, 10)


        # Split the data into training and testing sets
        labelTrainX, labelTestX, labelTrainY, labelTestY = train_test_split(
            data[featureCols], data[targetTask], test_size=0.2, random_state=1)  # 80% training and 20% test

        self.callClassifier(classificationMethod, labelTrainX, labelTrainY, labelTestX, labelTestY)



    #region Analysis functions

    def recursiveFeatureElimination(self, data, target, featureCols, maxScore):
        """Performs recursive feature analysis and removes any features above the given score"""
        st = time.time()

        x = data[featureCols]
        y = data[target]

        # Use a decision tree model
        model = DecisionTreeClassifier()
        # Feed the decision tree estimator to the RFE function to determine the most important features
        selector = RFE(model, step=1)
        # Fit it to our data frame
        selector.fit(x, y)

        for feature in featureCols:
            if selector.ranking_[featureCols.index(feature)] > maxScore:
                #print(f"Removing feature {feature} with score {selector.ranking_[featureCols.index(feature)]}")
                data = data.drop(feature, axis=1)
                featureCols.remove(feature)

        et = time.time()
        print("Time elapsed during RFE: ", et - st)
        self.modelName += "RFE-"

    def linearRegressionAnalysis(self, data, target, featureCols):
        """Performs linear regression analysis on the given data and removes the least relevant half of the features"""
        st = time.time()

        x = data[featureCols]
        y = data[target]

        # Use linear regression model
        model = LinearRegression()
        # Train the model
        model.fit(x, y)
        # Get the coefficients
        coefficients = model.coef_
        avg = numpy.average(coefficients)

        for feature in featureCols:
            if coefficients[featureCols.index(feature)] < avg:
                #print(f"Removing feature {feature} with coefficient {coefficients[featureCols.index(feature)]}")
                data = data.drop(feature, axis=1)
                featureCols.remove(feature)

        et = time.time()
        print("Time elapsed LR: ", et - st)
        self.modelName += "LR-"

    def principalComponentAnalysis(self, data, target, featureCols, numComponents=10):
        """Performs principal component analysis on the given data and removes every feature except the most relevant"""
        st = time.time()

        x = data[featureCols]
        y = data[target]

        # Use PCA model
        model = PCA(numComponents)
        # Train and scale the model
        x_scaled = StandardScaler().fit_transform(x)
        pcaFeatures = model.fit_transform(x_scaled)

        featureCols = ['pc' + str(i) for i in range(1, numComponents + 1)]

        pcaDf = pandas.DataFrame(data=pcaFeatures, columns=featureCols)
        pcaDf[target] = y

        data = pcaDf

        et = time.time()
        print("Time elapsed PCA: ", et - st)


        trainX, testX, trainY, testY = train_test_split(
             data[featureCols], data[target], test_size=0.2, random_state=1)  # 80% training and 20% test
        self.modelName += "PCA-"
        self.callClassifier(sys.argv[2], trainX, trainY, testX, testY)

    #endregion


    #region Classification

    def callClassifier(self, classifier, trainX, trainY, testX, testY):
        """Calls the given classifier function with the given data"""
        print(f"Calling classifier {classifier}")
        if (classifier == 'dtc'):
            self.decisionTreeClassify(trainX, trainY, testX, testY)
        elif (classifier == 'lrc'):
            self.logisticRegressionClassify(trainX, trainY, testX, testY)
        elif (classifier == 'svc'):
            self.SupportVectorClassify(trainX, trainY, testX, testY)
        else:
            print("Invalid classifier")

    def decisionTreeClassify(self, x, y, testX, testY ):
        """Classify the data"""
        st = time.time()

        # Create Decision Tree classifer object
        clf = DecisionTreeClassifier()
        # # Train Decision Tree Classifer
        clf = clf.fit(x, y)
        # Predict the response for test dataset
        prediction = clf.predict(testX)
        # Model Accuracy, how often is the classifier correct?

        print("Accuracy:", metrics.accuracy_score(testY, prediction))
        print(metrics.classification_report(testY, prediction))
        et = time.time()
        print("Time elapsed DTC: ", et - st)

        # Save the model
        self.modelName += "DTC.sav"
        pickle.dump(clf, open(self.modelName, 'wb'))


    def logisticRegressionClassify(self, x, y, testX, testY):
        """Classify the data using linear regression"""
        st = time.time()

        # Create logistic regression classifier object
        reg = LogisticRegression()
        # Adjust the model
        reg.fit(x, y)
        # Classification report
        prediction = reg.predict(testX)

        print("Accuracy:", metrics.accuracy_score(testY, prediction))
        print(metrics.classification_report(testY, prediction))
        et = time.time()
        print("Time elapsed LRC: ", et - st)

        # Save the model
        self.modelName += "LRC.sav"
        pickle.dump(reg, open(self.modelName, 'wb'))

    def SupportVectorClassify(self, x, y, testX, testY):
        """Classify the data using a perceptron classifier"""
        st = time.time()

        n_estimators = 10
        clf = OneVsRestClassifier(BaggingClassifier(LinearSVC(dual=False), max_samples=1.0 / n_estimators,
                                                        n_estimators=n_estimators))
        clf.fit(x, y)
        prediction = clf.predict(testX)

        print("Accuracy:", metrics.accuracy_score(testY, prediction))
        print(metrics.classification_report(testY, prediction))
        et = time.time()
        print("Time elapsed SVC: ", et - st)

        # Save the model
        self.modelName += "SVC.sav"
        pickle.dump(clf, open(self.modelName, 'wb'))

    #endregion

if (__name__ == "__main__"):
    dataFile = sys.argv[1]
    classMethod = sys.argv[2]
    task = sys.argv[3]
    model = sys.argv[4] if len(sys.argv) > 4 else None

    NIDS(dataFile, classMethod, task, model)