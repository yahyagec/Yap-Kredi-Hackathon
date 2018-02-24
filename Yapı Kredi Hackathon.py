import pandas as pd
import numpy as np

        
def PassOneMonth(dataset1 , dataset2 , person , isPaid ,presentAmount , t, n):

    X = dataset1.iloc[:, 1:].values
    
    time_sample = np.zeros(len(X))    
    n_sample = np.zeros(len(X))
    
    for i in range(0 , len(X)):
        time_sample[i] = int((np.random.rand() + 1)*10)
        n_sample[i] = np.random.randint(time_sample[i]) + 1
    
    sample = np.random.rand(len(X))
    y = dataset1.iloc[:, 0].values
    X[: , 9] = -1*X[: , 9]
    
    
    ## Monthly payment apiden gelecek
    
    from sklearn.preprocessing import Imputer
    imputer = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
    imputer = imputer.fit(X[:, [4,5,9]])
    X[:, [4,5,9]] = imputer.transform(X[:, [4,5,9]])
    
    
    from sklearn.cross_validation import train_test_split
    X_train , X_test , y_train , y_test = train_test_split(X , y ,test_size = 0.2 , random_state = 0)
    
    
    # Fitting Multiple Linear Regression to the Training set
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression(n_jobs = -1)
    regressor.fit(X_train, y_train)
    
    
    # Predicting the Test set results
    
    y_pred = regressor.predict(X_test)
    
    
    
    #from sklearn.metrics import r2_score
    #r2score = r2_score(y_test, y_pred)
    
    y_pred = y_pred * 10
    
    for i in range(0 , len(y_pred)):
        if (y_pred[i] < 0):
            y_pred[i] = 0
    
    
    ###############################################################################
    
    ## Api dönülünce son iki sütun güncellenecek!
    
    X1 = dataset2.iloc[: , 1:19].values
    y1 = dataset2.iloc[: , 0].values
    
    # Encoding categorical data for X1 # 0,2,5,6,7
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    labelencoder_X = LabelEncoder()
    X1[:, 1] = labelencoder_X.fit_transform(X1[:,1])
    X1[:, 4] = labelencoder_X.fit_transform(X1[:,4])
    X1[:, 5] = labelencoder_X.fit_transform(X1[:,5])
    X1[:, 6] = labelencoder_X.fit_transform(X1[:,6])
    onehotencoder = OneHotEncoder(categorical_features = [1,4,5,6])
    X1 = onehotencoder.fit_transform(X1).toarray()
    
    labelencoder_y = LabelEncoder()
    y1 = labelencoder_y.fit_transform(y1)
    
    
    for i in range(0 , len(X1)):
        X1[i , 28] = X[i , 10]
        X1[i , 29] = X[i , 11]
        X1[i , 30] = y_pred[i]    
        
    X1_train , X1_test , y1_train , y1_test = train_test_split(X1 , y1 ,test_size = 0.2 , random_state = 0)
    
    # Fitting Multiple Linear Regression to the Training set
    from sklearn.linear_model import LinearRegression
    regressor2 = LinearRegression(n_jobs = -1)
    regressor2.fit(X1_train, y1_train)
    
    
    # Predicting the Test set results
    y1_pred = regressor2.predict(X1_test)

    y1_pred_format = regressor2.predict(X1)        
        
    
    interest_rate_yearly = 0.05 + np.exp(y1_pred[person])/100    
    interest_rate_monthly = np.power(interest_rate_yearly + 1,1/12)-1
    interest_rate_monthly_normal = 0.05 + np.exp(0)/100
    
    for i in range(0 , len(y1_pred)):
        if(i != person):
            interest_rate_yearly = 0.05 + np.exp(y1_pred[i])/100    
            interest_rate_monthly = np.power(interest_rate_yearly + 1,1/12)-1
            amount_min = (presentAmount * interest_rate_monthly * np.power(1 + interest_rate_monthly , (time_sample[i] - n)))/(np.power(1 + interest_rate_monthly , (time_sample[i] - n))-1)
            amount_normal = (presentAmount * interest_rate_monthly_normal  * np.power(1 + interest_rate_monthly_normal , time_sample[i]))/(np.power(1 + interest_rate_monthly_normal , time_sample[i])-1)
            X[i , 12] = (X[i , 12] + amount_min + (np.log(y1_pred[i]) * (amount_normal - amount_min)))/2
    
    amount_min_specific_person = (presentAmount * interest_rate_monthly * np.power(1 + interest_rate_monthly , (t - n)))/(np.power(1 + interest_rate_monthly , (t - n))-1)
    
    amount_normal_specific_person = (presentAmount * interest_rate_monthly_normal  * np.power(1 + interest_rate_monthly_normal , t))/(np.power(1 + interest_rate_monthly_normal , t)-1)
    
    suggestedAmount_specific_person = amount_min_specific_person + (np.log(y_pred[person]) * (amount_normal_specific_person - amount_min_specific_person))
    
    for i in range(0 , len(X)):
        if(sample[i] > 0.5 and i != person):
            X[i , 10] = X[i , 10] + 1
        elif (sample[i] < 0.5 and i != person):
            X[i , 11] = X[i , 11] - 1
    
    
    for i in range(0 , len(X1)):
        X1[i , 30] = y_pred[i]
    
    
    if(isPaid == 1):
        X[person , 10] = X[person , 10] + 1
    elif (isPaid == 0):
        X[person , 11] = X[person , 11] - 1    
        

    dataset1['IsPaid'] = X[: , 10]
    dataset1['IsNotPaid'] = X[: , 11]   
    dataset1['SuggestedAmount'] = X[: , 12]
    
    dataset2['Status'] =  y1_pred_format
    dataset2['IsPaid'] = X1[: , 28]
    dataset2['IsNotPaid'] = X1[: , 29]
    dataset2['RiskScore'] = X1[: , 30]
    
    # isPaid , t ,n API den gelecek
    
    return y_pred , suggestedAmount_specific_person , interest_rate_yearly , dataset1 , dataset2

# isPaid , t ,n API den gelecek

# y_pred UI a döndürülecek

t = 0
n = 0
isPaid = 1 
person = 0
presentAmount = 0
dataset1 = pd.read_csv('Rough Data 2.csv')
dataset2 = pd.read_csv('Rough Data.csv')
dataset1 = dataset1.drop('Unnamed: 0', 1)

PassOneMonth(dataset1 , dataset2 , person , isPaid , presentAmount ,t , n)







    







