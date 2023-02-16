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

        print(data.isnull().sum())

        # Make a new dataframe with only the rows that have an attack_cat (drop all rows where atkCat is null)
        atkCatData = data
        atkCatData.dropna(axis='rows', subset=['attack_cat'], inplace=True)


        x = data[featureCols]                   # Features
        yLabel = data['Label']                  # Target variable
        xAtkCat = atkCatData[featureCols]       # Features
        yAtkCat = atkCatData['attack_cat']      # Target variable

        trainX, testX, trainY, testY = train_test_split(x, yLabel, test_size=0.2, random_state=1) # 80% training and 20% test
        catTrainX, catTestX, catTrainY, catTestY = train_test_split(xAtkCat, yAtkCat, test_size=0.2,
                                                        random_state=1)  # 80% training and 20% test

        # Create Decision Tree classifer object
        clf = DecisionTreeClassifier()
        atkCatClf = DecisionTreeClassifier()

        # # Train Decision Tree Classifer
        clf = clf.fit(trainX, trainY)
        atkCatClf = atkCatClf.fit(catTrainX, catTrainY)

        #Predict the response for test dataset
        prediction = clf.predict(testX)
        atkCatPrediction = atkCatClf.predict(catTestX)

        # Model Accuracy, how often is the classifier correct?
        print("Accuracy:",metrics.accuracy_score(testY, prediction))
        print(metrics.classification_report(testY, prediction))

        # Model Accuracy, how often is the classifier correct?
        print("CAT Accuracy:",metrics.accuracy_score(catTestY, atkCatPrediction))
        print(metrics.classification_report(catTestY, atkCatPrediction))



if (__name__ == "__main__"):
    NIDS()