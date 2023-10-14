#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df1 = pd.read_csv('TelcomCustomer-Churn_1.csv')
df1.head()


# In[3]:


df2 = pd.read_csv('TelcomCustomer-Churn_2.csv')
df2.head()


# In[4]:


print(df1.shape)
print(df2.shape)


# In[5]:


df1.dtypes


# In[6]:


df2.dtypes


# In[7]:


df4 = pd.merge(df1,df2, on ='customerID')
df4.head()


# In[8]:


df4.shape


# In[9]:


df4.dtypes


# In[10]:


print(df1.isin(df4))
print(df2.isin(df4))


# In[11]:


print(df1[~df1['customerID'].isin(df4['customerID'])])
print(df2[~df2['customerID'].isin(df4['customerID'])])


# In[12]:


Percentage_Missing = df4.isnull().sum()* 100 / len(df4)
missing_value_df =pd.DataFrame({'column_name': df4.columns, 'percent_missing' : Percentage_Missing})
missing_value_df


# In[13]:


print(df4.customerID.value_counts())
print(df4.gender.value_counts())
print(df4.SeniorCitizen.value_counts())
print(df4.Partner.value_counts())
print(df4.Dependents.value_counts())
print(df4.tenure.value_counts())
print(df4.PhoneService.value_counts())
print(df4.MultipleLines.value_counts())
print(df4.InternetService.value_counts())
print(df4.OnlineSecurity.value_counts())
print(df4.OnlineBackup.value_counts())
print(df4.DeviceProtection.value_counts())
print(df4.TechSupport.value_counts())
print(df4.StreamingTV.value_counts())
print(df4.StreamingMovies.value_counts())
print(df4.Contract.value_counts())
print(df4.PaperlessBilling.value_counts())
print(df4.PaymentMethod.value_counts())
print(df4.MonthlyCharges.value_counts())
print(df4.TotalCharges.value_counts())
print(df4.Churn.value_counts())


# In[14]:


df4.info()


# In[15]:


df4['TotalCharges'] = pd.to_numeric(df4['TotalCharges'],errors='coerce')


# In[16]:


df4.info()


# In[17]:


for feature in df4.columns: # Loop through all columns in the dataframe
    if df4[feature].dtype == 'object': # Only apply for columns with categorical strings
        df4[feature] = pd.Categorical(df4[feature])# Replace strings with an integer
df4.head(10)


# In[18]:


df4.info()


# In[19]:


import seaborn as sb


# In[20]:


df4['customerID'].unique


# In[21]:


def CategoricalVariableDivision(df4):
    #Replacing categorical values with 0 and 1 for easy standardization
    replaceStruct = {"Partner": {"No": 0, "Yes": 1 },"Churn":{"No": 0, "Yes": 1 },"Dependents": {"No": 0, "Yes": 1 },
                 "PaperlessBilling": {"No": 0, "Yes": 1 },"Contract": {"Month-to-month": 0, "One year": 1 ,"Two year":2}
                    }
    #Applying one hot encoding on categorical variables with unique values
    oneHotCols=["gender","DeviceProtection","PhoneService","TechSupport","StreamingTV","StreamingMovies","MultipleLines","InternetService","OnlineSecurity","PaymentMethod","OnlineBackup"]
    df4=df4.replace(replaceStruct)
    df4=pd.get_dummies(df4, columns=oneHotCols)
    #Dropping off unique and unnecessary columns
    df4=df4.drop(["customerID","gender_Male","PhoneService_No","MultipleLines_No phone service","InternetService_No","OnlineSecurity_No internet service","PaymentMethod_Mailed check","OnlineBackup_No internet service","DeviceProtection_No internet service","TechSupport_No internet service","StreamingMovies_No internet service","StreamingTV_No internet service"],axis=1)
    return df4

    


# In[22]:


df4 = CategoricalVariableDivision(df4)


# In[23]:


def clean_dataset(df4):
    assert isinstance(df4, pd.DataFrame), "df4 needs to be a pd.DataFrame"
    df4.dropna(inplace=True)
    indices_to_keep = ~df4.isin([np.nan, np.inf, -np.inf]).any(axis=1)
    return df4[indices_to_keep].astype(np.float64)


# In[24]:


clean_dataset(df4)


# In[25]:


X = df4.drop("Churn" , axis=1)
y = df4.pop("Churn")


# In[26]:


df5 = pd.concat([df4,y],axis =1)
df5.info()


# In[27]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=1)


# In[28]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train[['tenure','MonthlyCharges','TotalCharges']]=scaler.fit_transform(X_train[['tenure','MonthlyCharges','TotalCharges']])


# In[29]:


from xgboost import XGBClassifier


# In[30]:


model = XGBClassifier()
model.fit(X_train, y_train)


# In[31]:


y_pred = model.predict(X_test)
predictions =[round(value) for value in y_pred]
predictions


# In[32]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,predictions)
accuracy


# In[33]:


from sklearn import metrics
cm=metrics.confusion_matrix(y_test, y_pred, labels=[0, 1])

df_cm = pd.DataFrame(cm, index = [i for i in ["No","Yes"]],
                  columns = [i for i in ["No","Yes"]])
plt.figure(figsize = (7,5))
sb.heatmap(df_cm, annot=True ,fmt='g')


# In[34]:


#BegginningOfPart2


# In[35]:


def LogisticRegression(X_train,y_train,X_test,y_test):
    from sklearn.impute import SimpleImputer
    rep_0 = SimpleImputer(missing_values=0, strategy="mean")
    cols=X_train.columns
    X_train = pd.DataFrame(rep_0.fit_transform(X_train))
    X_test = pd.DataFrame(rep_0.fit_transform(X_test))
    from sklearn import metrics
    from sklearn.linear_model import LogisticRegression
    # Fit the model on train
    model = LogisticRegression(solver="liblinear")
    model.fit(X_train, y_train)
    #predict on test
    y_predict = model.predict(X_test)
    model_score = model.score(X_test, y_test)
    print("Logistic Regression Model score is",model_score)
    print("Confusion Matrix")
    cm=metrics.confusion_matrix(y_test, y_predict, labels=[1, 0])
    df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],
                  columns = [i for i in ["Predict 1","Predict 0"]])
    plt.figure(figsize = (7,5))
    sb.heatmap(df_cm, annot=True)
    
   
    
    return model_score













# In[36]:


def KNN(X_train,y_train,X_test,y_test):
    from sklearn.neighbors import KNeighborsClassifier
    from scipy.stats import zscore
    NNH = KNeighborsClassifier(n_neighbors= 5, weights ='distance')
    NNH.fit(X_train, y_train)
    predicted_labels= NNH.predict(X_test)
    model_score = NNH.score(X_test, y_test)
    print("KNN Model score is",model_score)
    
    


# In[37]:


def SVM(X_train,y_train,X_test,y_test):
    from sklearn import svm
   
    clf = svm.SVC(gamma=0.025, C=3)  
    clf.fit(X_train , y_train)
    y_pred = clf.predict(X_test)
    model_score = clf.score(X_test, y_test)
    print("SVM Model score is",model_score)
    print("Confusion Matrix")
    from sklearn import metrics
    cm=metrics.confusion_matrix(y_test, y_pred, labels=[1, 0])
    df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],
                  columns = [i for i in ["Predict 1","Predict 0"]])
    plt.figure(figsize = (7,5))
    sb.heatmap(df_cm, annot=True)
    return model_score


# In[38]:


def Naive_Bayes(X_train,y_train,X_test,y_test):
    from sklearn.naive_bayes import GaussianNB # using Gaussian algorithm from Naive Bayes
    # create the model
    diab_model = GaussianNB()

    diab_model.fit(X_train, y_train)
    diab_test_predict = diab_model.predict(X_test)
    from sklearn import metrics
    #print("Naive_Bayes Model Accuracy: {0:.4f}".format(metrics.accuracy_score(y_test, diab_test_predict)))
    model_score= metrics.accuracy_score(y_test, diab_test_predict)
    print("Naive_Bayes score is",model_score)
    print("Confusion Matrix")
    cm=metrics.confusion_matrix(y_test, diab_test_predict, labels=[1, 0])
    df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],
                  columns = [i for i in ["Predict 1","Predict 0"]])
    plt.figure(figsize = (7,5))
    sb.heatmap(df_cm, annot=True)
    

    return model_score


# In[39]:


def DecisionTree(X_train,y_train,X_test,y_test):
    from sklearn.tree import DecisionTreeClassifier
    dTree = DecisionTreeClassifier(criterion = 'gini', random_state=1)
    dTree.fit(X_train, y_train)
    model_score = dTree.score(X_test, y_test)
    print("Decision Tree Model score is",model_score)
    from sklearn.metrics import confusion_matrix
    y_predict = dTree.predict(X_test)
    cm = confusion_matrix(y_test, y_predict, labels=[1, 0])
    df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],
                  columns = [i for i in ["Predict 1","Predict 0"]])
    plt.figure(figsize = (7,5))
    sb.heatmap(df_cm, annot=True ,fmt='g')
    
    return model_score


# In[40]:


X_train.shape


# In[41]:


X_test.shape


# In[42]:


y_train.shape


# In[43]:


y_test.shape


# In[44]:


KNN(X_train,y_train,X_test,y_test)


# In[45]:


def Ensemble(x):
    if(x==1):
        LogisticRegression(X_train,y_train,X_test,y_test)
        
    elif(x==2):
        KNN(X_train,y_train,X_test,y_test)
        
    elif(x==3):
        SVM(X_train,y_train,X_test,y_test)
        
    elif(x==4):
        Naive_Bayes(X_train,y_train,X_test,y_test)
        
    elif(x==5):
        DecisionTree(X_train,y_train,X_test,y_test)
        
    elif(x==6):
        LogisticRegression(X_train,y_train,X_test,y_test)
        KNN(X_train,y_train,X_test,y_test)
        SVM(X_train,y_train,X_test,y_test)
        Naive_Bayes(X_train,y_train,X_test,y_test)
        DecisionTree(X_train,y_train,X_test,y_test)
        
    else:
        print("Invalid Input")
        
        
        
        
        
        


# In[ ]:





# In[47]:


LogisticRegression(X_train,y_train,X_test,y_test)
KNN(X_train,y_train,X_test,y_test)
SVM(X_train,y_train,X_test,y_test)
Naive_Bayes(X_train,y_train,X_test,y_test)
DecisionTree(X_train,y_train,X_test,y_test)


# In[ ]:





# In[ ]:





# In[ ]:




