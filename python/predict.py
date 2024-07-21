from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from joblib import load
import pandas as pd
import numpy as np
import addCols
import training

def makepredict_exp(logfile, loaded_model_file):
    df = pd.read_csv(logfile, names=addCols.add_cols())
    df['Attack Type'] = df.target.apply(lambda r:addCols.attackTypes()[r[:-1]])

    #Drop Target and Attack Type
    df = df.drop(['target', ], axis = 1)

    # flag feature mapping 
    fmap = {'SF':0, 'S0':1, 'REJ':2, 'RSTR':3, 'RSTO':4, 'SH':5, 'S1':6, 'S2':7, 'RSTOS0':8, 'S3':9, 'OTH':10} 
    df['flag'] = df['flag'].map(fmap)

    pmap = {'icmp':0, 'tcp':1, 'udp':2} 
    df['protocol_type'] = df['protocol_type'].map(pmap)
    df.drop('service', axis = 1, inplace = True)
    df.drop('num_root',axis = 1,inplace = True)

    #This variable is highly correlated with serror_rate and should be ignored for analysis.
    #(Correlation = 0.9983615072725952)
    df.drop('srv_serror_rate',axis = 1,inplace = True)

    #This variable is highly correlated with rerror_rate and should be ignored for analysis.
    #(Correlation = 0.9947309539817937)
    df.drop('srv_rerror_rate',axis = 1, inplace=True)

    #This variable is highly correlated with srv_serror_rate and should be ignored for analysis.
    #(Correlation = 0.9993041091850098)
    df.drop('dst_host_srv_serror_rate',axis = 1, inplace=True)

    #This variable is highly correlated with rerror_rate and should be ignored for analysis.
    #(Correlation = 0.9869947924956001)
    df.drop('dst_host_serror_rate',axis = 1, inplace=True)

    #This variable is highly correlated with srv_rerror_rate and should be ignored for analysis.
    #(Correlation = 0.9821663427308375)
    df.drop('dst_host_rerror_rate',axis = 1, inplace=True)

    #This variable is highly correlated with rerror_rate and should be ignored for analysis.
    #(Correlation = 0.9851995540751249)
    df.drop('dst_host_srv_rerror_rate',axis = 1, inplace=True)

    #This variable is highly correlated with dst_host_srv_count and should be ignored for analysis.
    #(Correlation = 0.9865705438845669)
    df.drop('dst_host_same_srv_rate',axis = 1, inplace=True)

    X = df.drop(['Attack Type', ], axis = 1)
    sc = MinMaxScaler()
    X = sc.fit_transform(X)
    model = load(loaded_model_file)
    predictions = model.predict(X)
    return predictions

def makepredict(logfile, loaded_model_file):
    output = []
    pred_array=[]
    clfd = load(loaded_model_file)
    file_in = open(logfile, 'r')
    for line in file_in.read().split('\n'):
        log_var_array=[]
        for log_var in line.split(','):
            var = float(log_var)
            log_var_array.append(var)
        pred_array.append(log_var_array)
    print(pred_array)
    prediction = clfd.predict(pred_array)
    print(prediction)
    for x in prediction:
        output.append(x)
    df = pd.DataFrame([["DOS", "10"], ["NORMAL", "0"]], columns=["Result", "Severity"])
    fig = df['Result'].value_counts().plot(kind="bar").get_figure()
    fig.savefig('test.png')
    with pd.ExcelWriter("predict.xlsx") as writer:
        df.to_excel(writer) 
    output.append("Exported predict.xlsx") 
    return output

def samplePredict():
    clfd=load('model.joblib')
    predict = clfd.predict([[0.00000000e+00, 0.00000000e+00, 
                             0.00000000e+00, 1.48837072e-06,0.00000000e+00, 
                             0.00000000e+00, 0.00000000e+00, 
                             0.00000000e+00, 0.00000000e+00, 
                             0.00000000e+00, 0.00000000e+00, 
                             0.00000000e+00,0.00000000e+00, 0.00000000e+00, 
                             0.00000000e+00, 0.00000000e+00,0.00000000e+00, 
                             0.00000000e+00, 1.00000000e+00, 1.00000000e+00, 
                             0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 
                             0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 
                             1.00000000e+00, 0.00000000e+00, 1.00000000e+00, 
                             0.00000000e+00]])
    print(predict)
    
# samplePredict() # Debug