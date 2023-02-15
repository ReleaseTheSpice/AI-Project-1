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
                    "ct_srv_src","ct_srv_dst","ct_dst_ltm","ct_src_","ltm","ct_src_dport_ltm","ct_dst_sport_ltm",
                    "ct_dst_src_ltm","attack_cat","Label"]

        # "self" is a reference to the current instance of the class
        self.data = pandas.read_csv("UNSW-NB15-BALANCED-TRAIN.csv", header=None, names=colNames)

        # same thing as above, but without label
        featureCols = ["srcip","sport","dstip","dsport","proto","state","dur","sbytes","dbytes","sttl","dttl","sloss",
                    "dloss","service","Sload","Dload","Spkts","Dpkts","swin","dwin","stcpb","dtcpb","smeansz","dmeansz",
                    "trans_depth","res_bdy_len","Sjit","Djit","Stime","Ltime","Sintpkt","Dintpkt","tcprtt","synack",
                    "ackdat","is_sm_ips_ports","ct_state_ttl","ct_flw_http_mthd","is_ftp_login","ct_ftp_cmd",
                    "ct_srv_src","ct_srv_dst","ct_dst_ltm","ct_src_","ltm","ct_src_dport_ltm","ct_dst_sport_ltm",
                    "ct_dst_src_ltm","attack_cat"]

        x = self.data[featureCols]  # Features
        y = self.data.Label         # Target variable

        trainX, testX, trainY, testY = train_test_split(x, y, test_size=0.2, random_state=1) # 80% training and 20% test

        # Create Decision Tree classifer object
        self.clf = DecisionTreeClassifier()

        # Encode categorical data
        ohe = preprocessing.OneHotEncoder(handle_unknown="ignore")

        # Fit the encoder to the data
        X = ohe.fit(trainX)

        # # Train Decision Tree Classifer
        # self.clf = self.clf.fit(trainX, trainY)

        #Predict the response for test dataset
        prediction = self.clf.predict(testX)

        # Model Accuracy, how often is the classifier correct?
        print("Accuracy:",metrics.accuracy_score(testY, prediction))
        print(metrics.classification_report(testY, prediction))



if (__name__ == "__main__"):
    NIDS()