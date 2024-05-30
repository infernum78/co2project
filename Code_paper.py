#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


# In[2]:


# Slope is end+1 to end-1
# Fluctuations counted only after 3 minutes
   # +20 ppm changes in span of 3 measurements (t - t-2)


# In[3]:


# CALCULATING PREDICTORS OF SINGLE DATASET

def predictor(arr,start=0,end=5*29): 
    #arr is input filename, time, and truth
    #start = number of measurement to start at, about 29 per min
    #end = number of measurements to stop at
    #constraint column = 1 means we found the range, 0 otherwise

    filename = arr[0];
    table = pd.read_csv(filename, skiprows=range(0,3), names=['Time','Temp','RH','CO2'])
    time = arr[1];
    truth = arr[2];
    
    assert len(time) == len(truth)+1
    assert list(table.columns) == ['Time', 'Temp', 'RH', 'CO2']

    #predictors
    net = []
    proportion = []
    ratio = []
    string = []
    slope = []
    net_hum = []
    net_temp = []
    fluct_co2 = []
    fluct_hum = []
    
    #other 
    time_str = []
    constraint = []
    
    for i in range(0,len(truth)): #finding df in specified time range
        t_one = time[i]+':'
        t_two = time[i+1]+':'
        f_one = (table['Time'].str.find(t_one))
        f_two = (table['Time'].str.find(t_two))
        ind_range = range(f_one[f_one>1].index[0],f_two[f_two>1].index[0])
        truth_val = truth[i]

        #finding df 
        df_total = table.iloc[ind_range,:]

        if start<0 or end>len(df_total):
            const = 0 
            # 0 is bad
            df = df_total 
        else:
            const = 1
            df = df_total.iloc[range(start,end),:]
        
        #adding stats to arrays, which will be returned in the form of a dataframe
        net.append(df.iloc[len(df)-1,3]-df.iloc[0,3])
        proportion.append(inc(df)/(len(df)))
        ratio.append(inc(df)/(len(df)-inc(df)))
        string.append(max_string(df))
        slope.append(slope_t(df,end))
        net_hum.append(df.iloc[len(df)-1,2]-df.iloc[0,2])
        net_temp.append(float(df.iloc[len(df)-1,1])-float(df.iloc[0,1]))
        fluct_co2.append(flucts(df,3))
        fluct_hum.append(flucts(df,2))
        
        time_str.append(time[i] + '-' + time[i+1])
        constraint.append(const)
    
    d = {'time':time_str,'net CO2':net,'proportion':proportion,'ratio':ratio,
         'string':string,'slope':slope,'net humidity':net_hum,'net temperature':net_temp,
         'fluct_CO2':fluct_co2,'fluct_hum':fluct_hum,
         'constraint':constraint,'truth':truth}
    predictors = pd.DataFrame(data = d)
    return predictors

def inc(df): #number of increases
    total = 0
    for i in range(len(df)-1):
        if (df.iloc[i+1,3]-df.iloc[i,3])>0:
            total += 1
    return total

def max_string(df): #max length of increases
    mx = 0
    total = 0
    for i in range(len(df)-1):  
        if (df.iloc[i+1,3]-df.iloc[i,3])>0:
            total += 1
        else: #not counting 0s as increases
            if total > mx:
                mx = total
            total = 0
    return mx

def prop_initial(df): #proportion of numbers greater than initial value
    total = 0
    for i in range(len(df)):
        if df.iloc[i,3]>df.iloc[0,3]:
            total += 1
    return total/len(df)

def slope_t(df,endpt): #avg slope across three measurements before given end point
    if endpt+1 > len(df):
        return (df.iloc[len(df)-1,3]-df.iloc[len(df)-3,3])/2
        #if there aren't enough measurements, eg. we want measurement 5*29 but only 3*29 measurements
    return (df.iloc[endpt,3]-df.iloc[endpt-2,3])/2
    #takes average of t+1 and t-1 measurement for slope

def flucts(df,col): #number of fluctuations
# col is which column to take in, 1 = temp, 2 = RH, 3 = CO2
    count = 0
    if len(df)<29*3:
        return 0
    if col==2:
    # ANOVA / histograms show diff=0.4, starting after 1.5 min is best
        for i in range(round(29*1.5),len(df)-2):  
            if (df.iloc[i+2,col]-df.iloc[i,col])>0.4:
                count += 1
    if col==3: 
    # ANOVA / histograms show diff=10, starting after 3 min is best
        for i in range(round(29*3),len(df)-2):  
            if (df.iloc[i+2,col]-df.iloc[i,col])>10:
                count += 1
    return count


# In[4]:


# TESTING DATA

time = ['10:50','10:57','11:04','11:10','11:16','11:22','11:30',
        '11:36','11:39','11:43','11:48','11:53','11:58','12:03',
        '12:08','12:13','12:18','12:23','12:25','12:30','12:35','12:40']
truth = [1,0,1,0,1,0,1,2,1,0,1,0,1,0,1,0,1,2,1,0,1]
ttt = ['data_zero.txt',time,truth]

predictor(ttt).head()


# In[5]:


# STREAMLINING PREDICTORS FOR MULTIPLE DATASETS

def all_predictor(arr_2d,start=0,end=5*29):
    #arr is the 2d array, [[txt_filename, time, truth],etc.]
    #start is which measurement to start, end is which measurement to end 
    all_data = pd.DataFrame()
    for i in range(len(arr_2d)):
        data = predictor(arr_2d[i],start=start,end=end)
        name = [(i+1) for j in range(len(data))]
        data.insert(len(data.columns)-2,'trial #',name)
        all_data = pd.concat([all_data,data])           
    return all_data 

def remove_bad(df): #Anything that isn't a test is removed, or has too few data points (constraint=0)
    return df[(df['truth']<2) & (df['constraint']==1)] 


# In[6]:


# TESTING DATA => PREDICTOR DATAFRAME

time_1 = ['12:44','12:50','12:55','12:58','13:03','13:08','13:12','13:17','13:20',
        '13:24','13:27','13:32','13:37','13:42','13:45','13:50','13:55','14:00','14:05',
        '14:08','14:13','14:18','14:20','14:25','14:30', #starting to face forward
        '14:35','14:40','14:43','14:48','14:53','14:58','15:03','15:05','15:10',
        '15:15','15:20','15:25']
truth_1 = [0,1,2,1,0,1,0,2,1,2,0,1,0,2,1,0,2,0,2,1,0,2,1,0,1,0,2,1,0,1,0,2,1,0,1,0]
ttt = [['data_zero.txt',time,truth],['data_one.txt',time_1,truth_1]]

df = all_predictor(ttt) 
df_final = remove_bad(df)
df_final.head()


# In[7]:


# CREATING REGRESSION MODELS / ACCURACY AND FALSE POSITIVE

def log_reg(predictors, cols):
    X = predictors.iloc[:,cols].to_numpy()
    y = predictors.iloc[:,len(predictors.columns)-1].to_numpy()
    model = LogisticRegression(fit_intercept=True, max_iter=300)
    model.fit(X,y)

    intercept = model.intercept_
    coeffs = model.coef_

    #unpacking intercept and coefficients
    for i in intercept:
        int_coef = [i]
    for j in coeffs:
        for k in j:
            int_coef.append(k)

    accuracy = acc(X, y, model)
    false_positive = fp(X, y, model)
    
    return int_coef, model, accuracy, false_positive

def acc(X, y, model): 
    y_hat = model.predict(X)
    length = len(y)
    correct = sum(y_hat==y)
    return correct/length

def fp(X, y, model): 
    y_hat = model.predict(X)
    total_neg = len(y) - sum(y)
    false_pos = 0
    for i in range(len(y_hat)):
        if y[i]==0 and y_hat[i]==1:
            false_pos += 1
    return false_pos/total_neg

# There is code that can help if we want to divide up our data into test and training sets
# but we don't have nearly enough data to do so. Removed test/training sets division. 


# In[8]:


# PREDICTOR DATAFRAME => REGRESSION MODEL 

int_coef, model, accuracy, false_positive = log_reg(df_final,[1,2])
int_coef, model, accuracy, false_positive


# In[9]:


# Steps to using code: (to get logistic regression for a singular time frame)
# 1. Data in 2D array to input into predictor() or all_predictor()
# 2. Use remove_bad() to remove incomplete data and put the column numbers in a variable
# 3. Use log_reg() to get int_coef, model, accuracy, fp_rate


# In[10]:


# FALSE POSITIVE / ACCURACY VS TIME GRAPHING 
# 1,2,3,4,5 minutes vs singular logistic regression 

# time is in minutes
def fp_graph(arr_2d, cols, time=np.linspace(1,5,5)): 
    # time interval from 1-5 minutes 
    fp_rate = []
    time = (time*29).round().astype(int)
    for t in time: 
        predictors = all_predictor(arr_2d,end=t)
        modified_predictors = remove_bad(predictors)
        int_coef, model, accuracy, false_positive = log_reg(modified_predictors, cols)
        fp_rate.append(false_positive)
    plt.plot(time/29,fp_rate,label='False Positive Rate')
    return fp_rate

def acc_graph(arr_2d, cols, time=np.linspace(1,5,5)):
    # time interval from 1-5 minutes 
    acc_rate = []
    time = (time*29).round().astype(int)
    for t in time: 
        predictors = all_predictor(arr_2d,end=t)
        modified_predictors = remove_bad(predictors)
        int_coef, model, accuracy, false_positive = log_reg(modified_predictors, cols)
        acc_rate.append(accuracy)
    plt.plot(time/29,acc_rate,label='Accuracy Rate')
    return acc_rate


# In[11]:


# fp_rate = fp_graph(ttt,[1,2])
# acc_rate = acc_graph(ttt, [1,2])
# plt.legend()


# In[12]:


# F-Statistic function / ANOVA analysis
# COL INPUT TO ROC_GRAPH AND LOG_REG_PREDICT NEEDS BRACKETS
    # because taking nonbracket results in series, which won't maintain the shape 
    # bracketed results in dataframe, which keeps the shape when transformed into 2D-array

from scipy.stats import f_oneway

def f_stat(data, col): 
    # data = dataframe, col = # of single column we want statistic from 
    r, c = data.shape
    df = data.iloc[:,[col,c-1]]
    df_t = df[df['truth']==1].iloc[:,0].to_numpy() # array with occupant
    df_f = df[df['truth']==0].iloc[:,0].to_numpy() # array with no occupant
    F, p_val = f_oneway(df_t, df_f)
    return F, p_val

def log_reg_predict(predictors, cols): #Gives the probability of each data occurrence
    X = predictors.iloc[:,cols].to_numpy()
    y = predictors.iloc[:,len(predictors.columns)-1].to_numpy()
    model = LogisticRegression(fit_intercept=True, max_iter=300)
    model.fit(X,y)
    y_pred = model.predict_proba(X)
    return y_pred 

def roc_graph(data, cols): 
    # data = dataframe, col = array of column #s we want model
    y_test = np.array(data['truth'])
    y_pred = np.array(log_reg_predict(data,cols)[:,1]) 
    # second column only, tells us chance of predicting a 1 (first col tells us chance of predicting 0)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plt.plot(fpr,tpr)
    plt.xlabel('FPR'), plt.ylabel('TPR') 
    score = str(round(roc_auc_score(y_test, y_pred),5))
    plt.title('AUC Score: ' + score)


# In[ ]:





# In[13]:


# ACTUAL DATA

#6/30
time_9 = ['5:56','6:01','6:06','6:09','6:14','6:19','6:22','6:27','6:32',
          '6:35','6:40','6:45','6:48','6:53','6:58','7:01','7:06','7:11',
          '7:14','7:19','7:24','7:35','7:40','7:45','7:48','7:53']
truth_9 = [1,0,2,1,0,2,1,0,2,1,0,2,1,0,2,1,0,2,1,0,2,1,0,2,1]

#7/1
time_10 = ['5:57','6:02','6:07','6:10','6:15','6:20','6:23','6:28','6:33',
           '6:36','6:41','6:46','6:49','6:54','6:59','7:02','7:07','7:12',
           '7:15','7:20','7:25','7:28','7:33','7:38','7:41','7:46','7:51',
           '7:54','7:59','8:04']
truth_10 = [1,0,2,1,0,2,1,0,2,1,0,2,1,0,2,1,0,2,1,0,2,1,0,2,1,0,2,1,0]

#7/6
time_11 = ['6:11','6:16','6:21','6:24','6:29','6:34','6:37','6:42','6:47',
           '6:50','6:55','7:00','7:03','7:08','7:13','7:16','7:21','7:26',
           '7:29','7:34','7:39','7:42','7:47','7:52','7:55','8:00','8:05',
           '8:08','8:13','8:18']
truth_11 = [1,0,2,1,0,2,1,0,2,1,0,2,1,0,2,1,0,2,1,0,2,1,0,2,1,0,2,1,0]

#7/7
time_12 = ['6:40','6:45','6:50','6:53','6:58','7:03','7:06','7:11','7:16',
           '7:19','7:24','7:29','7:32','7:37','7:42']
truth_12 = [1,0,2,1,0,2,1,0,2,1,0,2,1,0]

#7/10
time_14 = ['6:46','6:51','6:56','6:59','7:04','7:09','7:12','7:17','7:22',
           '7:25','7:30','7:35','7:42','7:47','7:52','7:55','8:00','8:05']
truth_14 = [1,0,2,1,0,2,1,0,2,1,0,2,1,0,2,1,0]

ttt = [['data_nine.txt',time_9,truth_9],['data_ten.txt',time_10,truth_10],
      ['data_eleven.txt',time_11,truth_11],['data_twelve.txt',time_12,truth_12],
      ['data_fourteen.txt',time_14,truth_14]]
df = all_predictor(ttt) 
df_final = remove_bad(df)
# pd.set_option('display.max_rows', None)
# df_final


# In[14]:


# int_coef, model, accuracy, false_positive = log_reg(df_final,[1,2,5])
# fp_rate = fp_graph(ttt,[1],time=np.linspace(1,5,9))
# acc_rate = acc_graph(ttt,[1],time=np.linspace(1,5,9))
# plt.legend()
# plt.xlabel('Time (minutes)')
# plt.ylabel('Accuracy / False Positive Rate')
# plt.title('Net Conc. Diff')


# In[15]:


# HISTOGRAM RESULTS FROM DATA 

st_5min = df_final[['net CO2','truth']]
st_5min_t = st_5min[st_5min['truth']==1]
st_5min_f = st_5min[st_5min['truth']==0]
plt.hist(st_5min_t['net CO2'],24,(-500,500),alpha=0.8,ec='black',label='Occupant')
plt.hist(st_5min_f['net CO2'],24,(-500,500),alpha=0.8,ec='black',label='No Occupant')
plt.legend()
plt.xlabel('Change in Net CO$_{2}$')
plt.ylabel('Count')
plt.xlim((-400,400))


# In[16]:


st_5min = df_final[['net humidity','truth']]
st_5min_t = st_5min[st_5min['truth']==1]
st_5min_f = st_5min[st_5min['truth']==0]
plt.hist(st_5min_t['net humidity'],24,(-4,4),alpha=0.8,ec='black',label='Occupant')
plt.hist(st_5min_f['net humidity'],24,(-4,4),alpha=0.8,ec='black',label='No Occupant')
plt.legend()
plt.xlabel('Change in Relative Humidity (%)')
plt.ylabel('Count')


# In[17]:


st_5min = df_final[['fluct_CO2','truth']]
st_5min_t = st_5min[st_5min['truth']==1]
st_5min_f = st_5min[st_5min['truth']==0]
plt.hist(st_5min_t['fluct_CO2'],24,(0,20),alpha=0.8,ec='black',label='Occupant')
plt.hist(st_5min_f['fluct_CO2'],24,(0,20),alpha=0.8,ec='black',label='No Occupant')
plt.legend()
plt.xlabel('fluct_CO2')


# In[ ]:


st_5min = df_final[['fluct_hum','truth']]
st_5min_t = st_5min[st_5min['truth']==1]
st_5min_f = st_5min[st_5min['truth']==0]
plt.hist(st_5min_t['fluct_hum'],24,(0,20),alpha=0.8,ec='black',label='Occupant')
plt.hist(st_5min_f['fluct_hum'],24,(0,20),alpha=0.8,ec='black',label='No Occupant')
plt.legend()
plt.xlabel('fluct_hum')


# In[19]:


# Higher change in CO2 is slightly associated with presence of occupant while more negative 
# change in CO2 is slightly associated with no occupant 

# Positive change RH is associated with presence of occupant while negative change RH is 
# associated with no occupant

# More fluctuations in both CO2 and RH is highly associated with presence of occupant


# In[20]:


row, col = df_final.shape
for i in range(1,col-3):
    f, pval = f_stat(df_final, i)
    name = df_final.columns[i]
    print(name + ': F-statistic = ' + str(f) + ', with p-val = ' + str(pval))


# In[21]:


int_coef, model, accuracy, false_positive = log_reg(df_final,[1,6,8,9]) 
accuracy
# highest accuracy using net CO2, net humidity, fluct CO2, fluct humidity


# In[22]:


int_coef, model, accuracy, false_positive = log_reg(df_final,[1,6,9])
accuracy


# In[23]:


int_coef, model, accuracy, false_positive = log_reg(df_final,[1,6,8])
accuracy


# In[24]:


int_coef, model, accuracy, false_positive = log_reg(df_final,[1,8,9])
accuracy


# In[25]:


# AUC SCORE OF BEST MODEL (net CO2, net humidity, fluctations CO2, fluctuations humidity

y_test = np.array(df_final['truth'])
y_pred = np.array(log_reg_predict(df_final,[1,6,8,9])[:,1]) # best model
# second column only, tells us chance of predicting a 1 (first col tells us chance of predicting 0)
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.plot(fpr,tpr)
plt.xlabel('False Positive Rate'), plt.ylabel('True Positive Rate') 
score = str(round(roc_auc_score(y_test, y_pred),3))
plt.title('AUC Score: ' + score)

def log_reg_predict_1(predictors, cols): # Gives the probability of each data occurrence
    X = predictors.iloc[:,cols].to_numpy()
    y = predictors.iloc[:,len(predictors.columns)-1].to_numpy()
    model = LogisticRegression(fit_intercept=True, max_iter=300)
    model.fit(X,y)
    return model

A = log_reg_predict_1(df_final,[1,6,8,9])
A.coef_, A.intercept_



