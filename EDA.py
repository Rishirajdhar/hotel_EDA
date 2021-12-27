#!/usr/bin/env python
# coding: utf-8

# In[1]:


#DATA Collection


# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns     #visualisation
import matplotlib.pyplot as plt         #visualisation
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.express as px
import folium
sns.set(color_codes=True)          #default colours are used

#for ML
from sklearn.model_selection import train_test_split, KFold, cross_validate, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import eli5   #feature import evaluation


# In[3]:


df = pd.read_csv(r"C:\Users\User\Downloads\hotel_bookings.csv\hotel_bookings.csv")


# In[4]:


df


# In[5]:


df.shape


# In[6]:


list(df.columns)


# In[7]:


fig, ax = plt.subplots(figsize=(12,6))
ax.hist(df['hotel'])
ax.set_xlabel('Hotel type')
ax.set_ylabel('Quantity-->')
plt.show()


# In[8]:


df.dtypes


# In[9]:


#Data cleaning and preparation


# In[10]:


df = df.drop(columns = ['arrival_date_year','agent','company','reservation_status_date'])
df.shape
df.head(5)


# In[11]:


#rows containing duplicate data
duplicate_rows_df = df[df.duplicated()]
print("Number of duplicate rows:",duplicate_rows_df.shape)


# In[12]:


df.count()


# In[13]:


#dropping the duplicates
df = df.drop_duplicates()
df.head(5)


# In[14]:


df.count()


# In[15]:


#finding the null values
print(df.isnull().sum())


# In[16]:


#dropping the missing values
df = df.dropna()             #will drop the not available values
df.count()


# In[17]:


#after dropping the values
print(df.isnull().sum())


# In[18]:


df.describe()


# In[19]:


fig, ax = plt.subplots(figsize = (12,6))
ax.hist(df['hotel'])
ax.set_xlabel('Hotel type')
ax.set_ylabel('Quantity-->')
plt.show()


# In[20]:


df.boxplot()
plt.xticks(rotation=90)
plt.figure(figsize=(20,15))


# In[21]:


#plotiing boxplot to find outliers
#plot for lead_time variable
plt.figure(figsize=(20,5))
sns.boxplot(x = df['lead_time'], color = 'Red')


# In[22]:


#plotting boxplot to find outliers
#plot for adr variable
plt.figure(figsize=(20,5))
sns.boxplot(x=df['adr'],color='Blue')


# In[23]:


#dropping outliers
q1 = df.quantile(0.25)
q3 = df.quantile(0.75)
IQR = q3-q1
print("\nInternalQuartileRange=\n",IQR)
df1 = df[~((df<(q1-1.5*IQR))|(df>(q3+1.5*IQR))).any(axis=1)]
df1.head(5)
df1.shape


# In[24]:


df.dtypes


# In[25]:


#Data Visualisation


# In[26]:


fig,ax = plt.subplots(figsize=(10,6))
#plt.subplots() is a function that returns a tuple containing a figure and axes object(s). Thus when using fig, ax = plt.subplots
ax.scatter(df['market_segment'],df['days_in_waiting_list'])
ax.set_xlabel('market segment')
ax.set_ylabel('days on waiting list')
plt.show()


# In[27]:


grp = df.groupby('arrival_date_month')
p=grp['lead_time'].agg(np.mean)
q=grp['stays_in_week_nights'].agg(np.mean)
r=grp['stays_in_weekend_nights'].agg(np.mean)
s=grp['booking_changes'].agg(np.sum)
t=grp['days_in_waiting_list'].agg(np.sum)
u=grp['adr'].agg(np.mean)
v=grp['hotel'].agg(np.sum)
print(p)
print(q)
print(r)
print(s)
print(t)
print(u)
print(v)


# In[28]:


#np.corrcoef(df['hotel'],df['arrival_date_month'])
import seaborn as sns
sns.heatmap(pd.crosstab(df.arrival_date_month,df.hotel),cmap="coolwarm")


# In[29]:


df1=df.groupby(['arrival_date_month','hotel']).size()
df1=df1.unstack()
df1.plot(kind='bar')


# In[30]:


df2=df.groupby(['is_canceled','hotel']).size()
df2=df2.unstack()
df2.plot(kind='bar')


# In[31]:


df3=df.groupby(['hotel','deposit_type']).size()
#df3=df3.unstack()
df3.plot(kind='bar')


# In[32]:


df4=df.groupby(['is_canceled','is_repeated_guest']).size()
#df4=df4.unstack()
df4.plot(kind='bar')


# In[33]:


#fig, ax = plt.subplots(figsize=(16,6))
df5=df.groupby(['hotel','market_segment']).size()
df5=df5.unstack()
df5.plot(kind='bar',figsize=(10,5))


# In[34]:


df6=df.groupby(['hotel','distribution_channel']).size()
df6=df6.unstack()
df6.plot(kind='bar',figsize=(10,5))


# In[35]:


dfshort = pd.DataFrame(df,columns = ['total_of_special_requests','babies','children','adults'])
corrMatrix = dfshort.corr()
print(corrMatrix)


# In[36]:


df7=df.groupby(['is_repeated_guest','customer_type']).size()
df7=df7.unstack()
df7.plot(kind='bar',figsize=(10,5))


# In[37]:


df8=df.groupby(['reservation_status','market_segment']).size()
df8=df8.unstack()
df8.plot(kind='bar',figsize=(10,5))


# In[38]:


plt.figure(figsize=(16,5))
plt.plot(p,'ro',color='r')
plt.xticks(rotation=90)
plt.title('Month of the year vs Lead time',fontsize=15)
plt.xlabel('Month of the Year')
plt.ylabel('Lead time')
plt.show()


# In[39]:


plt.figure(figsize=(16,5))
plt.plot(q,'ro',color='r')
plt.xticks(rotation=90)
plt.title('Month of the year vs Stays on week nights',fontsize=15)
plt.xlabel('Month of the Year')
plt.ylabel('Week night')
plt.show()


# In[40]:


plt.figure(figsize=(16,5))
plt.plot(r,'ro',color='r')
plt.xticks(rotation=90)
plt.title('Month of the year vs Stays on weekend nights',fontsize=15)
plt.xlabel('Month of the Year')
plt.ylabel('Week nights')
plt.show()


# In[41]:


plt.figure(figsize=(16,5))
plt.plot(s,'ro',color='r')
plt.xticks(rotation=90)
plt.title('Month of the year vs Booking changes',fontsize=15)
plt.xlabel('Month of the Year')
plt.ylabel('Booking_changes')
plt.show()


# In[42]:


plt.figure(figsize=(16,5))
plt.plot(t,'ro',color='r')
plt.xticks(rotation=90)
plt.title('Month of the year vs Days in waiting list',fontsize=15)
plt.xlabel('Month of the Year')
plt.ylabel('Days in waiting list')
plt.show()


# In[43]:


#heatmap
#Finding the relations between the variables
plt.figure(figsize=(20,10))
c = df.corr()
sns.heatmap(c,cmap="coolwarm",annot=True) #BrBG, RdGy, coolwarm
c


# In[44]:


#Data modeling


# In[45]:


df.shape


# In[46]:


total_cancelations = df["is_canceled"].sum()
print(total_cancelations)
rh_cancelations = df.loc[df["hotel"] == "Resort Hotel"]["is_canceled"].sum()
print(rh_cancelations)
ch_cancelations = df.loc[df["hotel"] == "City Hotel"]["is_canceled"].sum()
print(ch_cancelations)


# In[47]:


#as percent
rel_cancel = total_cancelations/df.shape[0]*100
rh_rel_cancel = rh_cancelations/df.loc[df["hotel"]== "Resort Hotel"].shape[0]*100
ch_rel_cancel = ch_cancelations/df.loc[df["hotel"]== "City Hotel"].shape[0]*100

print(f"Total bookings canceled: {total_cancelations:,} ({rel_cancel:.0f} %)")
print(f"Resort hotel bookings canceled: {rh_cancelations:,} ({rh_rel_cancel:.0f} %)")
print(f"City hotel bookings canceled: {ch_cancelations:,} ({ch_rel_cancel:.0f} %)")


# In[48]:


# create a DataFrame with the relevant data:

res_book_per_month = df.loc[(df["hotel"] == "Resort Hotel")].groupby("arrival_date_month")["hotel"].count()
res_cancel_per_month = df.loc[(df["hotel"] == "Resort Hotel")].groupby("arrival_date_month")["is_canceled"].sum()

cty_book_per_month = df.loc[(df["hotel"] == "City Hotel")].groupby("arrival_date_month")["hotel"].count()
cty_cancel_per_month = df.loc[(df["hotel"] == "City Hotel")].groupby("arrival_date_month")["is_canceled"].sum()

res_cancel_data = pd.DataFrame({"Hotel": "Resort Hotel",
                                "Month": list(res_book_per_month.index),
                                "Bookings": list(res_book_per_month.values),
                                "Cancelations": list(res_cancel_per_month.values)})
cty_cancel_data = pd.DataFrame({"Hotel": "City Hotel",
                                "Month": list(cty_book_per_month.index),
                                "Bookings": list(cty_book_per_month.values),
                                "Cancelations": list(cty_cancel_per_month.values)})

full_cancel_data = pd.concat([res_cancel_data, cty_cancel_data], ignore_index=True)
full_cancel_data["cancel_percent"] = full_cancel_data["Cancelations"]/full_cancel_data["Bookings"]*100
                                                      
                                                      
#order by month:
ordered_months=["January","February","March","April","May","June","July","August","September","October","November","December"]
full_cancel_data["Month"] = pd.Categorical(full_cancel_data["Month"],categories = ordered_months,ordered=True)
                                                      
#Show figure:
plt.figure(figsize =(12,8))
sns.barplot(x = "Month", y = "cancel_percent", hue = "Hotel",
           hue_order = ["City Hotel", "Resort Hotel"], data = full_cancel_data)
plt.title("Cancelations per month", fontsize = 16)
plt.xlabel("Month", fontsize = 16)
plt.xticks(rotation = 45)
plt.ylabel("Cancelations [%]", fontsize = 16)
plt.legend(loc="upper right")
plt.show()                       


# In[49]:


cancel_corr = df.corr()["is_canceled"]
cancel_corr.abs().sort_values(ascending = False)[1:]


# In[50]:


df.groupby("is_canceled")["reservation_status"].value_counts()


# In[51]:


#manually choose columns to include

num_features = ["lead_time","arrival_date_week_number","arrival_date_day_of_month","stays_in_weekend_nights",
                "stays_in_week_nights",
               "adults","children","babies","is_repeated_guest","previous_cancellations","previous_bookings_not_canceled",
                "required_car_parking_spaces","total_of_special_requests","adr"]
cat_features = ["hotel", "arrival_date_month","meal","market_segment",
               "distribution_channel","reserved_room_type","deposit_type","customer_type"]

#separate features and predicted value
features = num_features + cat_features
x = df.drop(["is_canceled"], axis=1)[features]
y=df["is_canceled"]
#preprocess numerical features:
num_transformer = SimpleImputer(strategy="constant")
#preprocessing categorical features:
cat_transformer = Pipeline(steps=[("imputer",SimpleImputer(strategy="constant",fill_value="Unknown")),
                                 ("onehot",OneHotEncoder(handle_unknown='ignore'))])
#Bundle preprocessing for numerical and categorical features:
preprocessor = ColumnTransformer(transformers=[("num",num_transformer,num_features,
                                               ("cat",cat_transformer,cat_features))])


# In[54]:


#define model
base_models = [("DT_model",DecisionTreeClassifier(random_state=42)),
              ("RF_model", RandomForestClassifier(random_state=42,n_jobs=-1))]
#split data into 'kfolds' parts for cross validation,
# use shuffle to ensure random distribution of data:
kfolds = 4#4 = 75% train, 25% validation
split=KFold(n_splits = kfolds, shuffle=True,random_state = 42)

# Preprocessing, fitting, making predictions and scoring for every model:
for name, model in base_models:
    #pack preprocessing of data and the model in a pipeline:
    model_steps = Pipeline(steps=[('preprocessor',preprocessor),('model',model)])
    #get cross validation score for each model:
    cv_results = cross_val_score(model_steps, x, y, cv=split, scoring="accuracy",n_jobs=-1)
    #output
    min_score = round(min(cv_results),4)
    max_score = round(max(cv_results),4)
    mean_score = round(np.mean(cv_results),4)
    std_dev = round(np.std(cv_results),4)
    print(f"{name} cross validation accuracy score: {mean_score} +/- {std_dev} (std) min: {min_score}, max: {max_score}")
    


# In[53]:


lead_cancel_data = df.groupby("lead_time")["is_canceled"].describe()
lead_cancel_data_10 = lead_cancel_data.loc[lead_cancel_data["count"] >=10]
plt.figure(figsize=(12,8))
sns.regplot(x=lead_cancel_data_10.index, y=lead_cancel_data_10["mean"].values*100)
plt.title("Effect of lead time on cancelation", fontsize=16)
plt.xlabel("Lead time", fontsize=16)
plt.ylabel("cancelations[%]",fontsize=16)
#plt.xlim(0,365)
plt.show()


# In[ ]:




