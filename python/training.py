
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns #used for heatmap
import time
import addCols
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import graphs
import preprocessing
from joblib import load, dump

def runTraining(logfile='', model_file=''):
    output = []
    if len(logfile)==0:
        df = pd.read_csv(preprocessing.kdd_preprocessing(10),names=addCols.add_cols())
    else:
        df = pd.read_csv(logfile,names=addCols.add_cols())
    print(df)
    output.append("Data Sample Before Preprocessing: \n")
    output.append(df)
    #Adding Attack Type column
    df['Attack Type'] = df.target.apply(lambda r:addCols.attackTypes()[r[:-1]])
    # df.head() ## Debug check dataframe

    #Shape of dataframe and getting data type of each feature
    df.shape
    output.append('Data Frame Shape: '+str(df.shape))
    df['target'].value_counts()
    df['Attack Type'].value_counts()
    print(df['Attack Type'].value_counts())
    # Code: Finding missing values of all features
    df.isnull().sum()

    
    # Finding categorical features 
    num_cols = df._get_numeric_data().columns 
    
    cate_cols = list(set(df.columns)-set(num_cols)) 
    cate_cols.remove('target') 
    cate_cols.remove('Attack Type') 
    
    cate_cols 

    #Data Correlation - Finding multicollinearity
    #Visualization
    def bar_graph(feature):
        fig = df[feature].value_counts().plot(kind="bar").get_figure()
        fig.savefig(feature+'.png')
    bar_graph('protocol_type')
    plt.figure(figsize=(15,3))
    bar_graph('service')
    bar_graph('flag')
    bar_graph('logged_in') # logged_in (1 if successfully logged in; 0 otherwise): We notice that just 70000 packets are successfully logged in.
    bar_graph('target')
    bar_graph('Attack Type')
    df.columns


    #Data Correlation
    #df = df.dropna('columns')# drop columns with NaN

    df = df[[col for col in df if df[col].nunique() > 1]]# keep columns where there are more than 1 unique values
    print("df = df[[col for col in df if df[col].nunique() > 1]]")
    df.shape
    #corr = df.corr()

    #plt.figure(figsize=(15,12))

    #sns.heatmap(corr)

    plt.show() # show graphs

    # Correlated Data / To Be Removed
    df['num_root'].corr(df['num_compromised'])
    df['srv_serror_rate'].corr(df['serror_rate'])
    df['srv_count'].corr(df['count'])
    df['srv_rerror_rate'].corr(df['rerror_rate'])
    df['dst_host_same_srv_rate'].corr(df['dst_host_srv_count'])
    df['dst_host_srv_serror_rate'].corr(df['dst_host_serror_rate'])
    df['dst_host_srv_rerror_rate'].corr(df['dst_host_rerror_rate'])
    df['dst_host_same_srv_rate'].corr(df['same_srv_rate'])
    df['dst_host_srv_count'].corr(df['same_srv_rate'])
    df['dst_host_same_src_port_rate'].corr(df['srv_count'])
    df['dst_host_serror_rate'].corr(df['serror_rate'])
    df['dst_host_srv_serror_rate'].corr(df['serror_rate'])
    df['dst_host_serror_rate'].corr(df['srv_serror_rate'])
    df['dst_host_srv_serror_rate'].corr(df['srv_serror_rate'])
    df['dst_host_rerror_rate'].corr(df['rerror_rate'])
    df['dst_host_rerror_rate'].corr(df['srv_rerror_rate'])
    df['dst_host_srv_rerror_rate'].corr(df['rerror_rate'])
    df['dst_host_srv_rerror_rate'].corr(df['srv_rerror_rate'])

    # Drop multicollinear variables
    array_to_drop = addCols.mc_var_remove()
    for var_to_drop in array_to_drop:
        df.drop(var_to_drop,axis = 1,inplace = True)


    # protocol_type feature mapping 
    pmap = {'icmp':0, 'tcp':1, 'udp':2} 
    df['protocol_type'] = df['protocol_type'].map(pmap)

    # flag feature mapping 
    fmap = {'SF':0, 'S0':1, 'REJ':2, 'RSTR':3, 'RSTO':4, 'SH':5, 'S1':6, 'S2':7, 'RSTOS0':8, 'S3':9, 'OTH':10} 
    df['flag'] = df['flag'].map(fmap) 

    df.drop('service', axis = 1, inplace = True) #Remove irrelevant features such as ‘service’ before modelling

    df.head()
    output.append('Removed Multicollinear elements. Sample with remaining variables: \n')
    output.append(df.head())
    df.shape
    df.columns

    #df_std = df.std()
    #df_std = df_std.sort_values(ascending = True)
    #df_std

    # Splitting the dataset 
    df = df.drop(['target', ], axis = 1) 
    
    print(df.shape) 
    output.append("Dataframe shape Before MinMax: "+str(df.shape))
    # Target variable and train set 
    y = df[['Attack Type']] 
    X = df.drop(['Attack Type', ], axis = 1) 
    
    output.append('MinMaxing Dataframe...')
    '''
    excelfile = 'df_format_before_minmax.xlsx'
    with pd.ExcelWriter(excelfile) as writer:
        df.head().to_excel(writer)
    output.append(f"Successfully exported {excelfile}")
    '''
    
    sc = MinMaxScaler() # Transform features by scaling each feature to a given range / standardizes data
    X = sc.fit_transform(X) # Fit to data, then transform it.

    
    # Split test and train data  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42) # 2/3 training data 1/3 test data
    print(X_train.shape, X_test.shape) 
    print(y_train.shape, y_test.shape)
    # #Decision Tree

    #Training
    train_time = 0
    if len(model_file) == 0: # if no model provided, train with provided data
        clfd = graphs.tree.DecisionTreeClassifier(criterion ="entropy", max_depth = 4)
        output.append("Training new Model based on given data")
        start_time = time.time()
        clfd.fit(X_train, y_train.values.ravel()) 
        end_time = time.time()
        train_time = end_time -start_time
    else: # if not, load model file
        clfd = load(model_file)
        output.append("Loading selected model...")
    if train_time ==0:
        print("Training time: N/A due to pretrained model") 
        output.append("Training time: N/A due to pretrained model")
    else:
        print("Training time: ", train_time) 
        output.append("Training time: " + str(train_time))

    #Testing
    start_time = time.time() 
    y_test_pred = clfd.predict(X_train) # ['dos', 'normal'] etc
    end_time = time.time() 
    test_time = str(end_time-start_time)
    print("Testing time: ", test_time) 
    output.append("Testing time: " + str(test_time))



    print("Done")
    if len(model_file) == 0: # if no model provided, return both output array and model dump
        return output, clfd
    else: # if not, load model file
        return output
    
    # graphs.graphTree(clfd, filename)

def run_debug():
    return "DEBUG TIME"
'''
def save_model(clfd):
    dump(clfd, 'model.joblib') # dump model
    load('model.joblib') # load model

    output = open("prediction.txt", "w")
    #pred = clfd.predict(test_packet)

    output.write(repr(X_train[1])+"\n")
    for x in y_test_pred:
        output.write(repr(x)+"\n")
    output.close'''
def select_model():
    model = preprocessing.etc.trim(preprocessing.etc.search_for_file_path("joblib"))
    return str(model)

def runPredictionTraining(logfile, model_file):
    output = []
    df = pd.read_csv(preprocessing.kdd_preprocessing(10),names=addCols.add_cols())
    df_prediction = pd.read_csv(logfile,names=addCols.add_cols())
    print(df)
    output.append("Data Sample Before Preprocessing: \n")
    output.append(df)
    #Adding Attack Type column
    df['Attack Type'] = df.target.apply(lambda r:addCols.attackTypes()[r[:-1]])
    # df.head() ## Debug check dataframe

    #Shape of dataframe and getting data type of each feature
    df.shape
    output.append('Data Frame Shape: '+str(df.shape))
    df['target'].value_counts()
    df['Attack Type'].value_counts()
    print(df['Attack Type'].value_counts())
    # Code: Finding missing values of all features
    df.isnull().sum()

    
    # Finding categorical features 
    num_cols = df._get_numeric_data().columns 
    
    cate_cols = list(set(df.columns)-set(num_cols)) 
    cate_cols.remove('target') 
    cate_cols.remove('Attack Type') 
    
    cate_cols 

    #Data Correlation - Finding multicollinearity
    #Visualization
    def bar_graph(feature):
        df[feature].value_counts().plot(kind="bar")
    bar_graph('protocol_type')
    plt.figure(figsize=(15,3))
    bar_graph('service')
    bar_graph('flag')
    bar_graph('logged_in') # logged_in (1 if successfully logged in; 0 otherwise): We notice that just 70000 packets are successfully logged in.
    bar_graph('target')
    bar_graph('Attack Type')
    df.columns


    #Data Correlation
    #df = df.dropna('columns')# drop columns with NaN

    df = df[[col for col in df if df[col].nunique() > 1]]# keep columns where there are more than 1 unique values
    print("df = df[[col for col in df if df[col].nunique() > 1]]")
    df.shape
    #corr = df.corr()

    #plt.figure(figsize=(15,12))

    #sns.heatmap(corr)

    # plt.show() # show graphs

    # Correlated Data / To Be Removed
    df['num_root'].corr(df['num_compromised'])
    df['srv_serror_rate'].corr(df['serror_rate'])
    df['srv_count'].corr(df['count'])
    df['srv_rerror_rate'].corr(df['rerror_rate'])
    df['dst_host_same_srv_rate'].corr(df['dst_host_srv_count'])
    df['dst_host_srv_serror_rate'].corr(df['dst_host_serror_rate'])
    df['dst_host_srv_rerror_rate'].corr(df['dst_host_rerror_rate'])
    df['dst_host_same_srv_rate'].corr(df['same_srv_rate'])
    df['dst_host_srv_count'].corr(df['same_srv_rate'])
    df['dst_host_same_src_port_rate'].corr(df['srv_count'])
    df['dst_host_serror_rate'].corr(df['serror_rate'])
    df['dst_host_srv_serror_rate'].corr(df['serror_rate'])
    df['dst_host_serror_rate'].corr(df['srv_serror_rate'])
    df['dst_host_srv_serror_rate'].corr(df['srv_serror_rate'])
    df['dst_host_rerror_rate'].corr(df['rerror_rate'])
    df['dst_host_rerror_rate'].corr(df['srv_rerror_rate'])
    df['dst_host_srv_rerror_rate'].corr(df['rerror_rate'])
    df['dst_host_srv_rerror_rate'].corr(df['srv_rerror_rate'])

    # Drop multicollinear variables
    array_to_drop = addCols.mc_var_remove()
    for var_to_drop in array_to_drop:
        df.drop(var_to_drop,axis = 1,inplace = True)


    # protocol_type feature mapping 
    pmap = {'icmp':0, 'tcp':1, 'udp':2} 
    df['protocol_type'] = df['protocol_type'].map(pmap)

    # flag feature mapping 
    fmap = {'SF':0, 'S0':1, 'REJ':2, 'RSTR':3, 'RSTO':4, 'SH':5, 'S1':6, 'S2':7, 'RSTOS0':8, 'S3':9, 'OTH':10} 
    df['flag'] = df['flag'].map(fmap) 

    df.drop('service', axis = 1, inplace = True) #Remove irrelevant features such as ‘service’ before modelling

    df.head()
    output.append('Removed Multicollinear elements. Sample with remaining variables: \n')
    output.append(df.head())
    df.shape
    df.columns

    #df_std = df.std()
    #df_std = df_std.sort_values(ascending = True)
    #df_std

    # Splitting the dataset 
    df = df.drop(['target', ], axis = 1) 
    
    print(df.shape) 
    output.append("Dataframe shape Before MinMax: "+str(df.shape))
    # Target variable and train set 
    y = df[['Attack Type']] 
    X = df.drop(['Attack Type', ], axis = 1) 
    
    output.append('MinMaxing Dataframe...')
    '''
    excelfile = 'df_format_before_minmax.xlsx'
    with pd.ExcelWriter(excelfile) as writer:
        df.head().to_excel(writer)
    output.append(f"Successfully exported {excelfile}")
    '''
    
    sc = MinMaxScaler() # Transform features by scaling each feature to a given range / standardizes data
    X = sc.fit_transform(X) # Fit to data, then transform it.


    
    # Split test and train data  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42) # 2/3 training data 1/3 test data
    print(X_train.shape, X_test.shape) 
    print(y_train.shape, y_test.shape)
    #Decision Tree

    #Training
    train_time = 0
    if len(model_file) == 0: # if no model provided, train with provided data
        clfd = graphs.tree.DecisionTreeClassifier(criterion ="entropy", max_depth = 4)
        output.append("Training new Model based on given data")
        start_time = time.time()
        clfd.fit(X_train, y_train.values.ravel()) 
        end_time = time.time()
        train_time = end_time -start_time
    else: # if not, load model file
        clfd = load(model_file)
        output.append("Loading selected model...")
    if train_time ==0:
        print("Training time: N/A due to pretrained model") 
        output.append("Training time: N/A due to pretrained model")
    else:
        print("Training time: ", train_time) 
        output.append("Training time: " + str(train_time))

    #Testing
    start_time = time.time() 
    y_test_pred = clfd.predict(X_train) 
    end_time = time.time() 
    test_time = str(end_time-start_time)
    print("Testing time: ", test_time) 
    output.append("Testing time: " + str(test_time))

    print("Done")
    if len(model_file) == 0: # if no model provided, return both output array and model dump
        return output, clfd
    else: # if not, load model file
        return output