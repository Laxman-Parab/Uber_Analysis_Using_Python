#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing Packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os


# In[2]:


#Importing Datasets

files = os.listdir(r'A:\DataScience Projects\UberData')[-7:]
files


# In[3]:


files.remove('uber-raw-data-janjune-15.csv')
files


# In[4]:


#Concatenate all the 6 datasets into a dataframe using for loop.

path = r'A:\DataScience Projects\UberData'
final = pd.DataFrame()
for file in files:
    df = pd.read_csv(path+'/'+file,encoding = 'utf-8')
    final = pd.concat([df,final])   ##Contains all the concatenated data


# In[5]:


final.shape


# ##### Lat : The latitude of the Uber pickup
# 
# ##### Lon : The longitude of the Uber pickup
# 
# ##### Base : The TLC base company code affiliated with the Uber pickup

# In[6]:


df = final.copy()
df.head()


# In[7]:


df.dtypes


# In[8]:


df['Date/Time']= pd.to_datetime(df['Date/Time'],format = '%m/%d/%Y %H:%M:%S')       
                                                  #Converting to Date time format.


# In[9]:


df.dtypes


# In[10]:


df.head()


# In[11]:


#Creating seprate columns in the dataset for weekday, month, day, hour and minute

df['Weekday']=df['Date/Time'].dt.day_name()
df['Month']=df['Date/Time'].dt.month
df['Day']= df['Date/Time'].dt.day
df['Hour']=df['Date/Time'].dt.hour
df['Minute'] = df['Date/Time'].dt.minute


# In[12]:


df.head()


# In[13]:


df.dtypes


# #### Analysis of Journey by Weekdays

# In[14]:


df['Weekday'].value_counts()


# In[15]:


df['Weekday'].value_counts().index


# In[16]:


sns.set_theme(style="whitegrid")
sns.barplot(x = df['Weekday'].value_counts().index,y = df['Weekday'].value_counts(),
            order = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'])
plt.figure(figsize=(7,6))


# In[17]:


px.bar(x = df['Weekday'].value_counts().index,y = df['Weekday'].value_counts())
            


# After observing the barplots, we can conclude that the rush in Uber is highest on Thursdays.

# #### Analysis of Journey by Hours.

# In[18]:


sns.set_theme(style="whitegrid")
sns.barplot(x = df['Hour'].value_counts().index,y = df['Hour'].value_counts()).set(title = 'Journeys per Hours')


# After observing the barplot, we can conclude that the passengers/rush in Uber is at peak during 5 pm.

# In[19]:


df['Month'].unique()


# In[20]:


month={9:'Sep',5:'May',6:'June',7:'July',8:'August',4:'April'}
for i in df['Month'].unique():
    plt.figure(figsize=(5,3))
    df[df['Month']==i]['Hour'].hist()
    plt.title('Histrogram of {}'.format(month[i]))


# For each month,we can see that the peak hours for rush is in the evening.

# ### Which month has maximum rides?

# In[21]:


import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot


# In[22]:


x = df.groupby('Month')['Hour'].count().index
y = df.groupby('Month')['Hour'].count()
x , y


# In[23]:


trace1 = go.Bar( 
        x = df.groupby('Month')['Hour'].count().index,
        y = df.groupby('Month')['Hour'].count(),
        name= 'Priority')
iplot([trace1])


# In[ ]:





# Therefore,we can clearly see that the 9th month i.e. Septmeber has most rides.

# ### Analysis of journey of each day

# In[24]:


plt.figure(figsize=(10,8))
plt.hist(df['Day'], bins=31,color = 'teal')
plt.xlabel('Date of the month')
plt.ylabel('Total Journeys')
plt.title('Journeys by Month Day')


# In[ ]:





# ### Total Rides month wise

# In[25]:


plt.figure(figsize = (20,8))
for i,Month in enumerate (df['Month'].unique(),1):
    plt.subplot(3,2,i)
    df_cut= df[df['Month'] == Month]
    plt.hist(df_cut['Day'])
    plt.xlabel('Days in months'.format(month))
    plt.ylabel('Total Rides')  


# In[26]:


month={9:'Sep',5:'May',6:'June',7:'July',8:'August',4:'April'}
for i in df['Month'].unique():
    plt.figure(figsize=(5,3))
    df[df['Month']==i]['Day'].hist()
    plt.title('Histrogram of {}'.format(month[i]))


# There are maximum rides at end of every month.

# ### Rush in hour

# In[27]:


plt.figure(figsize= (20,10))
sns.set_style(style='whitegrid')
sns.pointplot(x='Hour',y= 'Lat',data=df,hue = 'Weekday',marker = '+').set_title('Hoursoffday vs Latiitide of Ride')
 
                                        ## This Pointplot give us idea of average latitudes.

                                    ## It means having more number of vehicle at this approximate latitude at this particular hour.


# ### Which base number gets popular by month name?
# 

# In[28]:


df.head()


# In[29]:


base = df.groupby(['Base','Month'])['Date/Time'].count().reset_index()


# In[30]:


base.head()


# In[31]:


plt.figure(figsize = (12,8))
sns.lineplot(x = 'Month',y = 'Date/Time',hue = 'Base',data = base)


# We can clearly see that the green line in the lineplot which represents 'B02617' base number gets popular by every month.

# ## Cross Analysis using heatmap 
# 

# In[32]:


def count_rows(rows):
    return len(rows)


# In[33]:


cross1 = df.groupby(['Weekday','Hour']).apply(count_rows)


# In[34]:


cross1


# In[35]:


pivot1 = cross1.unstack()


# In[36]:


pivot1


# In[37]:


plt.figure(figsize = (12,7))
sns.heatmap(pivot1,cmap = 'plasma')           ###    Heatmap by Hour and week day


# Most of the days the rides are more at highest during the day and eventually decrease by night.

# In[38]:


def hm(col1,col2):
    '''
    Generate a heatmap of the two specified columns
    '''
    cross1 = df.groupby([col1,col2]).apply(count_rows) 
    pivot1 = cross1.unstack()
    plt.figure(figsize = (12,7))
    return (sns.heatmap(pivot1,cmap = 'plasma'))         


# In[39]:


hm('Month','Hour')                        #Heatmap by month and hour


# Rides were at peak during evening hours of September.

# In[40]:


help(hm)


# In[41]:


hm('Month','Weekday')                    #Heatmap by month and Weekday


# In[42]:


hm('Month','Day')


# We observe that the number of trips increases each month, we can say that from April to September 2014, Uber was in a continuous improvement process.

# ### Analysis of location data points.
# 

# In[43]:


plt.figure(figsize=(10,6))

plt.plot(df['Lon'], df['Lat'],'r+', ms=1)
plt.xlim(-74.2, -73.7)
plt.ylim(40.6,41)


# From the above heatmap we can see that Midtown Manhattan is the big spot of Uber rides.

# #### Spatial Analysis

# In[44]:


df_sun = df[df['Weekday']=='Sunday']
df_sun.head()


# In[45]:


df_sun.shape


# In[46]:


rush = df_sun.groupby(['Lat','Lon'])['Weekday'].count().reset_index()


# In[47]:


rush.head()


# In[48]:


rush.columns = ['Lat','Lon','No. of Trips']


# In[49]:


rush.head()


# In[50]:


import folium as fy
from folium.plugins import HeatMap


# In[51]:


basemap = fy.Map()
basemap


# ### Spatial Analysis

# In[52]:


HeatMap(rush,radius = 20,zoom = 15,blur = 20).add_to(basemap)


# In[53]:


basemap


# #### Automate the analysis

# In[54]:


def dayhm(df,day):
    basemap = fy.Map()
    df1 = df[df['Weekday']==day]
    HeatMap(df1.groupby(['Lat','Lon'])['Weekday'].count().reset_index(),radius = 20,zoom = 15,blur = 20).add_to(basemap)
    return (basemap)


# In[55]:


dayhm(df,'Monday')


# In[56]:


uber_15=pd.read_csv(r'A:\DataScience Projects\UberData\uber-raw-data-janjune-15.csv')


# In[57]:


uber_15.head()


# In[58]:


uber_15.dtypes


# In[59]:


uber_15['Pickup_Date']= pd.to_datetime(uber_15['Pickup_date'],format = '%Y-%m-%d %H:%M:%S')


# In[60]:


uber_15.dtypes


# In[61]:


uber_15['Weekday'] = uber_15['Pickup_Date'].dt.day_name()
uber_15['Month'] = uber_15['Pickup_Date'].dt.month
uber_15['Day'] = uber_15['Pickup_Date'].dt.day
uber_15['Hour'] = uber_15['Pickup_Date'].dt.hour
uber_15['Minute'] = uber_15['Pickup_Date'].dt.minute


# In[62]:


uber_15.head()


# #### Uber pickups by month in NYC

# In[63]:


uber_15['Month'].value_counts()


# In[64]:


sns.barplot(y = uber_15['Month'].value_counts(),x = uber_15['Month'].value_counts().index)


# We can conclude that June had the most pickups. 

# ### Rush in NYC in hours

# In[65]:


sns.countplot(uber_15['Hour'],palette = 'muted')
plt.figure(figsize = (10,8))


# There is a significant increase demand in the pickups at evening

# #### Rush in NYC in day per hours

# In[66]:


dph = uber_15.groupby(['Weekday','Hour'])['Pickup_Date'].count().reset_index()


# In[67]:


dph.head()


# In[68]:


plt.figure(figsize = (12,8))
sns.pointplot(x = 'Hour',y = 'Pickup_Date',data = dph,hue = 'Weekday')


# During weekdays there is more rush in the morning compared to weekends.

# In[69]:


uber_foil = pd.read_csv(r'A:\DataScience Projects\UberData\Uber-Jan-Feb-FOIL.csv')


# In[70]:


uber_foil.head()


# Which base number has the most active vehicles?

# In[71]:


uber_foil['dispatching_base_number'].unique()


# In[72]:


sns.boxplot(x = 'dispatching_base_number',y = 'active_vehicles',data = uber_foil)


# Base number B02764 has the most active vehicles.

# Which base number has the most trips?

# In[73]:


sns.boxplot(x = 'dispatching_base_number',y = 'trips',data = uber_foil)


# Base number B02764 has the most trips.
How average trips/vehicles increase and decerase with dates with each base number.
# In[74]:


uber_foil['trips/vehicles'] = uber_foil['trips']/uber_foil['active_vehicles']


# In[75]:


uber_foil.head()


# In[76]:


uber_foil.set_index('date').groupby(['dispatching_base_number'])['trips/vehicles'].plot()
plt.ylabel('Average trips/vehicle')
plt.title('Demand vs Supply chart (Date-wise)')
plt.legend()
plt.figure(figsize=(12,8))


# B02764 & B02598 performs better
