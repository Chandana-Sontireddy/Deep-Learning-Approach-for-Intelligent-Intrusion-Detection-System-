import numpy as np 
import pandas as pd 
from sklearn import *
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
from sklearn import svm
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.ensemble import RandomForestClassifier
from sklearn_extensions.extreme_learning_machines.elm import GenELMClassifier
from sklearn_extensions.extreme_learning_machines.random_layer import RBFRandomLayer, MLPRandomLayer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

global labels 
global columns
global balance_data

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def preprocess(): 
    global labels
    global columns
    columns = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]

    labels = {"normal":0,"neptune":1,"warezclient":2,"ipsweep":3,"portsweep":4,"teardrop":5,"nmap":6,"satan":7,"smurf":8,"pod":9,"back":10,"guess_passwd":11,"ftp_write":12,"multihop":13,"rootkit":14,"buffer_overflow":15,"imap":16,"warezmaster":17,"phf":18,"land":19,"loadmodule":20,"spy":21,"perl":22,"saint":23,"mscan":24,"apache2":25,"snmpgetattack":26,"processtable":27,"httptunnel":28,"ps":29,"snmpguess":30,"mailbomb":31,"named":32,"sendmail":33,"xterm":34,"worm":35,"xlock":36,"xsnoop":37,"sqlattack":38,"udpstorm":39}
    balance_data = pd.read_csv("dataset.txt")
    dataset = ''
    index = 0
    cols = ''
    for index, row in balance_data.iterrows():
      for i in range(0,42):
        if(isfloat(row[i])):
          dataset+=str(row[i])+','
          if index == 0:
            cols+=columns[i]+','
      dataset+=str(labels.get(row[41]))
      if index == 0:
        cols+='Label'
      dataset+='\n'
      index = 1;
    
    f = open("clean.txt", "w")
    f.write(cols+"\n"+dataset)
    f.close()
    
def importdata(): 
    global balance_data
    balance_data = pd.read_csv("clean.txt") 
    # Printing the dataswet shape 
    print ("Dataset Lenght: ", len(balance_data)) 
    print ("Dataset Shape: ", balance_data.shape) 
    # Printing the dataset obseravtions 
    print ("Dataset: ",balance_data.head()) 
    return balance_data 

def splitdataset(balance_data): 
    X = balance_data.values[:, 0:37] 
    Y = balance_data.values[:, 38] 
    # Spliting the dataset into train and test 
    X_train, X_test, y_train, y_test = train_test_split( 
    X, Y, test_size = 0.2, random_state = 0)
    return X, Y, X_train, X_test, y_train, y_test 

# Function to perform training with giniIndex. 
def train_using_gini(X_train, X_test, y_train): 

	# Creating the classifier object 
	clf_gini = svm.SVC(C=2.0,gamma='scale',kernel = 'rbf', random_state = 2) 

	# Performing training 
	clf_gini.fit(X_train, y_train) 
	return clf_gini 
	
def elm(X_train, X_test, y_train):
    srhl_tanh = MLPRandomLayer(n_hidden=8, activation_func='tanh')
    cls = GenELMClassifier(hidden_layer=srhl_tanh)
    cls.fit(X_train, y_train)
    return cls

def randomForest(X_train, X_test, y_train):
    cls = RandomForestClassifier(n_estimators=1,max_depth=0.9,random_state=None) 
    cls.fit(X_train, y_train) 
    return cls 

def elmFeatureSelection(X_train, X_test, y_train):
    srhl_tanh = MLPRandomLayer(n_hidden=15, activation_func='tanh')
    cls = GenELMClassifier(hidden_layer=srhl_tanh)
    print('Original features:', X_train.shape[1])
    total = X_train.shape[1];
    X_train = SelectKBest(chi2, k=1).fit_transform(X_train, y_train)
    print('features set reduce after applying features concept:', (total - X_train.shape[1]))
    cls.fit(X_train, y_train)
    return cls

# Function to make predictions 
def prediction(X_test, clf_object): 
    # Predicton on test with giniIndex 
    y_pred = clf_object.predict(X_test) 
    print("Predicted values:") 
    print(y_pred) 
    #for i in range(len(X_test)):
      #print("X=%s, Predicted=%s" % (X_test[i], y_pred[i]))
    return y_pred 
	
# Function to calculate accuracy 
def cal_accuracy(y_test, y_pred): 
    print("Confusion Matrix: ", confusion_matrix(y_test, y_pred))
    print ("Accuracy : ", accuracy_score(y_test,y_pred)*100) 
    print("Report : ", classification_report(y_test, y_pred)) 


def main():
    #preprocess()
    data = importdata()
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data)
    clf_gini = train_using_gini(X_train, X_test, y_train) 
    #print("Results Using Gini Index:") 
    y_pred_gini = prediction(X_test, clf_gini) 
    cal_accuracy(y_test, y_pred_gini) 


    clf_gini = elm(X_train, X_test, y_train) 
    #print("Results Using Gini Index:") 
    y_pred_gini = prediction(X_test, clf_gini) 
    cal_accuracy(y_test, y_pred_gini) 

    clf_gini = randomForest(X_train, X_test, y_train) 
    #print("Results Using Gini Index:") 
    y_pred_gini = prediction(X_test, clf_gini) 
    cal_accuracy(y_test, y_pred_gini) 

    clf_gini = elmFeatureSelection(X_train, X_test, y_train) 
    #print("Results Using Gini Index:") 
    X_test = SelectKBest(chi2, k=1).fit_transform(X_test,y_test)
    
    y_pred_gini = prediction(X_test, clf_gini) 
    cal_accuracy(y_test, y_pred_gini) 
	
	

# Calling main function 
if __name__=="__main__": 
	main() 