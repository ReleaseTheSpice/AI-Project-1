###############################################################
### Martin Van Laethem  - Aiden Lonquist    - Cole Atkinson ###
### A01189452           - A01166561         - A00000000     ###
###                 AI Project 1 - NIDS                     ###
###############################################################

import pandas
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from sklearn import metrics # is used to create classification results
from sklearn import preprocessing # is used to encode the data
from sklearn.tree import export_graphviz # is used for plotting the decision tree
from six import StringIO # is used for plotting the decision tree
from IPython.display import Image # is used for plotting the decision tree
from IPython.core.display import HTML # is used for showing the confusion matrix
import pydotplus # is used for plotting the decision tree
from sklearn.svm import SVR
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy
# from correlation import uhhhh

class NIDS:
    """class for the network intrusion detection system"""

    def __init__(self):
        """Constructor"""
        colNames = ["srcip","sport","dstip","dsport","proto","state","dur","sbytes","dbytes","sttl","dttl","sloss",
                    "dloss","service","Sload","Dload","Spkts","Dpkts","swin","dwin","stcpb","dtcpb","smeansz","dmeansz",
                    "trans_depth","res_bdy_len","Sjit","Djit","Stime","Ltime","Sintpkt","Dintpkt","tcprtt","synack",
                    "ackdat","is_sm_ips_ports","ct_state_ttl","ct_flw_http_mthd","is_ftp_login","ct_ftp_cmd",
                    "ct_srv_src","ct_srv_dst","ct_dst_ltm","ct_src_ltm","ct_src_dport_ltm","ct_dst_sport_ltm",
                    "ct_dst_src_ltm","attack_cat","Label"]

        data = pandas.read_csv("UNSW-NB15-BALANCED-TRAIN.csv", names=colNames, skiprows=1)

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

        # Make a new dataframe for predicting the attack category
        # It has only the rows that have an attack_cat (drop all rows where atkCat is null)
        atkCatData = data
        atkCatData.dropna(axis='rows', subset=['attack_cat'], inplace=True)

        # data = recursiveFeatureElimination(data, 'Label', featureCols, 10)
        # atkCatData = recursiveFeatureElimination(atkCatData, 'attack_cat', featureCols, 10)

        # data = linearRegressionAnalysis(data, 'Label', featureCols)
        # atkCatData = linearRegressionAnalysis(atkCatData, 'attack_cat', featureCols)

        # data = principalComponentAnalysis(data, 'Label', featureCols, 10)
        # atkCatData = principalComponentAnalysis(atkCatData, 'attack_cat', featureCols, 10)


        # Split the data into training and testing sets
        labelTrainX, labelTestX, labelTrainY, labelTestY = train_test_split(
            data[featureCols], data['Label'], test_size=0.2, random_state=1)  # 80% training and 20% test
        catTrainX, catTestX, catTrainY, catTestY = train_test_split(
            atkCatData[featureCols], atkCatData['attack_cat'], test_size=0.2, random_state=1)  # 80% training and 20% test

        decisionTreeClassify(labelTrainX, labelTrainY, labelTestX, labelTestY)
        decisionTreeClassify(catTrainX, catTrainY, catTestX, catTestY)


#region Analysis functions

def recursiveFeatureElimination(data, target, featureCols, maxScore):
    """Performs recursive feature analysis and removes any features above the given score"""
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

def linearRegressionAnalysis(data, target, featureCols):
    """Performs linear regression analysis on the given data and removes the least relevant half of the features"""
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

def principalComponentAnalysis(data, target, featureCols, numComponents=10):
    """Performs principal component analysis on the given data and removes every feature except the most relevant"""
    x = data[featureCols]
    y = data[target]

    # Use PCA model
    model = PCA(numComponents)
    # Train and scale the model
    x_scaled = StandardScaler().fit_transform(x)
    pcaFeatures = model.fit_transform(x_scaled)

    # # Bar plot of explained_variance
    # plt.bar(
    #     range(1, len(model.explained_variance_) + 1),
    #     model.explained_variance_
    # )
    #
    # plt.xlabel('PCA Feature')
    # plt.ylabel('Explained variance')
    # plt.title('Feature Explained Variance')
    # plt.show()

    featureCols = ['pc' + str(i) for i in range(1, numComponents + 1)]

    pcaDf = pandas.DataFrame(data=pcaFeatures, columns=featureCols)
    pcaDf[target] = y

    data = pcaDf

    trainX, testX, trainY, testY = train_test_split(
         data[featureCols], data[target], test_size=0.2, random_state=1)  # 80% training and 20% test
    decisionTreeClassify(trainX, trainY, testX, testY)


#endregion


#region Classification functions

def decisionTreeClassify(x, y, testX, testY ):
    """Classify the data"""
    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier()
    # # Train Decision Tree Classifer
    clf = clf.fit(x, y)
    # Predict the response for test dataset
    prediction = clf.predict(testX)
    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:", metrics.accuracy_score(testY, prediction))
    print(metrics.classification_report(testY, prediction))


#endregion

if (__name__ == "__main__"):
    NIDS()