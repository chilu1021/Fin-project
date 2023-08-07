#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


# In[2]:


call = pd.read_csv("dataScienceChallenge_callLog.csv")
repayment = pd.read_csv("dataScienceChallenge_repayment.csv")


# In[3]:


call


# In[4]:


calls = call[pd.to_datetime(call['local_timestamp']) < pd.Timestamp(2021,1,1)]


# In[5]:


# # Calls data set with 6 columns. 
# Direction can be converted to binary variable, 
# local_timestamp can be seperated to different parts.


# In[6]:


calls


# In[7]:


# Repayment dataset has four columns. Person_id is both the key and foreign variable connects to the "call" dataset. 
# Paid_first_loan is the primrary determinator of the result.


# In[8]:


repayment


# In[9]:


calls.head()


# In[10]:


repayment.head()


# In[11]:


# Convert the name into ones that are more easy to read and code
calls = calls.rename(columns={'person_id_random' :'id', 'phone_randomized' : 'phone', 'contact_name_redacted' : 'name',
                              'local_timestamp' : 'timestamp'})
repayment = repayment.rename(columns={'person_id_random' :'id'})


# In[12]:


calls


# In[13]:


repayment


# In[14]:


# Checking the nulls in the calls dataset. Only the name column has nulls, which takes about 20% of the whole dataset. 
# According to the description, it might be caused by the issue of display on the user's phone 
missing_calls = calls.isnull()
missing_calls_count = missing_calls.sum()
missing_calls_count


# In[15]:


# calls.dropna(inplace=True)


# In[16]:


# No null values
missing_repayment = repayment.isnull()
missing_repayment_count = missing_repayment.sum()
missing_repayment_count


# In[17]:


# There are over ten thousands calls showing under one id, which is considered as abnormal. Therefore, new id is assigned later.
calls_num = calls['id'].value_counts()
calls['calls_num'] = calls['id'].map(calls_num) 


# In[18]:


calls


# In[19]:


# Convert direction into digital variables; 0 as missed, 1 as outgoing, 2 as incoming
calls['direction'] = calls['direction'].replace({'missed': 0, 'outgoing': 1, 'incoming': 2, 'unknown': 3})


# In[20]:


# Adding columns with incoming results 
calls_incoming = calls[calls['direction'] == 2].groupby('id').size()
calls_incoming_map = calls_incoming.to_dict()
calls['Incoming'] = calls['id'].map(calls_incoming_map) / calls['calls_num']

calls_outgoing = calls[calls['direction'] == 1].groupby('id').size()
calls_outgoing_map = calls_outgoing.to_dict()
calls['Outgoing'] = calls['id'].map(calls_outgoing_map) / calls['calls_num']

calls_missed = calls[calls['direction'] == 0].groupby('id').size()
calls_missed_map = calls_missed.to_dict()
calls['Missed'] = calls['id'].map(calls_missed_map) / calls['calls_num']


# In[21]:


# Replacing direction with duration equals to zero to missed
# calls.loc[calls['duration'] == 0, 'direction'] = 'missed' 


# In[22]:


# New id to identity the customer by the phone number and name
calls['caller_id'] = calls['phone'] + ['_'] + calls['name']


# In[23]:


# Add an call_id as a new identity to verify who does the user contacts to
calls_uni_num = calls['caller_id'].value_counts()
calls_num_map = calls_uni_num.to_dict()
calls['caller_num'] = calls['caller_id'].map(calls_num_map)


# In[24]:


# Add in the maximum and mean call duration to the table. Min is usually 0 so it was omitted
AVG_uni_DUR = calls.groupby('caller_id')['duration'].mean()
MAX_uni_DUR = calls.groupby('caller_id')['duration'].max()
MIN_uni_DUR = calls.groupby('caller_id')['duration'].min()
calls['AVG_caller_DUR'] = calls['caller_id'].map(AVG_uni_DUR.to_dict())
calls['MAX_caller_DUR'] = calls['caller_id'].map(MAX_uni_DUR.to_dict())


# In[25]:


# Add in the maximum and mean call duration to the table. Min is usually 0 so it was omitted
AVG_DUR = calls.groupby('id')['duration'].mean()
MAX_DUR = calls.groupby('id')['duration'].max()
MIN_DUR = calls.groupby('id')['duration'].min()
calls['AVG_DUR'] = calls['id'].map(AVG_DUR.to_dict())
calls['MAX_DUR'] = calls['id'].map(MAX_DUR.to_dict())


# In[26]:


# Add in a column for each id on the days of difference between timestamps
calls['timestamp'] = pd.to_datetime(calls['timestamp'])
time_length = calls.groupby('id')['timestamp'].max() - calls.groupby('id')['timestamp'].min()
day_dif = time_length.dt.days
calls['day_dif'] = calls['id'].map(day_dif)


# In[27]:


# Add in a column about the average amount of calls each user call each day
calls['phone_per_day'] = calls['calls_num'] / calls['day_dif']


# In[28]:


# Add in a column about the average amount of calls each user call to specific person each day
calls['phone_per_day_caller'] = calls['caller_num']/calls['day_dif']
calls['caller_percentage'] = calls['caller_num'] / calls['calls_num']


# In[29]:


def assign_relation(ratio):
    if ratio < 0.05:
        return 'Normal'
    elif ratio < 0.2:
        return 'Close'
    else:
        return 'Very Close'


# In[30]:


calls['Relation'] = calls['caller_percentage'].apply(assign_relation)


# In[31]:


calls[calls['id'] == 7477014136]['Relation']


# In[32]:


def assign_time_zone(time):
    if time.hour < 5:
        return 'Night'
    elif time.hour < 12:
        return 'Morning'
    elif time.hour < 17:
        return 'Afternoon'
    elif time.hour < 21:
        return 'Evening'
    else: 
        return 'Night'


# In[33]:


# Divide the timestamp into year, month, time
calls['year'] = calls['timestamp'].dt.year
calls['month'] = calls['timestamp'].dt.month
calls['day_of_week'] = calls['timestamp'].dt.dayofweek
calls['day_type'] = calls['day_of_week'].apply(lambda x: "Weekday" if x < 5 else "Weekend")
calls['time'] = calls['timestamp'].dt.time
calls['part_of_day'] = calls['time'].apply(assign_time_zone)


# In[34]:


# Merge two tables to include paid_first_loan column 
pay_map = repayment['paid_first_loan'].to_dict()
merged_table = pd.merge(calls, repayment, on = 'id', how = 'left')
merged_table


# In[35]:


new_repayment = repayment
new_repayment = new_repayment.merge(merged_table[['id', 'calls_num', 'AVG_DUR', 'MAX_DUR', 'day_dif', 'phone_per_day', 'Incoming', 'Outgoing']], 
                                    on='id', 
                                    how='left').drop_duplicates(subset='id')
new_repayment.dropna(inplace=True)
# print(calls[calls['id'] == 4856300649])
# new_repayment


# In[36]:


Weekday = calls[calls['day_type'] == 'Weekday'].groupby('id').size()
Weekday_map = Weekday.to_dict()
new_repayment['Weekday_num'] = new_repayment['id'].map(Weekday_map) / new_repayment['calls_num']


# In[37]:


Weekend = calls[calls['day_type'] == 'Weekend'].groupby('id').size()
Weekend_map = Weekend.to_dict()
new_repayment['Weekend_num'] = new_repayment['id'].map(Weekend_map) / new_repayment['calls_num']


# In[38]:


Morning = calls[calls['part_of_day'] == 'Morning'].groupby('id').size()
Morning_map = Morning.to_dict()
new_repayment['Morning_ratio'] = new_repayment['id'].map(Morning_map) / new_repayment['calls_num']


# In[39]:


Afternoon = calls[calls['part_of_day'] == 'Afternoon'].groupby('id').size()
Afternoon_map = Afternoon.to_dict()
new_repayment['Afternoon_ratio'] = new_repayment['id'].map(Afternoon_map) / new_repayment['calls_num']


# In[40]:


Evening = calls[calls['part_of_day'] == 'Evening'].groupby('id').size()
Evening_map = Evening.to_dict()
new_repayment['Evening_ratio'] = new_repayment['id'].map(Evening_map) / new_repayment['calls_num']


# In[41]:


night = calls[calls['part_of_day'] == 'Night'].groupby('id').size()
night_map = night.to_dict()
new_repayment['Night_ratio'] = new_repayment['id'].map(night_map) / new_repayment['calls_num']


# In[42]:


normal = calls[calls['Relation'] == 'Normal'].groupby('id').size()
normal_map = normal.to_dict()
new_repayment['Normal'] = new_repayment['id'].map(normal_map) / new_repayment['calls_num']


# In[43]:


close = calls[calls['Relation'] == 'Close'].groupby('id').size()
close_map = close.to_dict()
new_repayment['Close'] = new_repayment['id'].map(close_map) / new_repayment['calls_num']


# In[44]:


closer = calls[calls['Relation'] == 'Very Close'].groupby('id').size()
closer_map = closer.to_dict()
new_repayment['Very Close'] = new_repayment['id'].map(closer_map) / new_repayment['calls_num']


# In[45]:


new_repayment['signup_date'] = pd.to_datetime(new_repayment['signup_date'])
new_repayment['disbursement_date'] = pd.to_datetime(new_repayment['disbursement_date'])
new_repayment


# # Visualization
# 

# In[46]:


# Target Variable Distribution
sns.set_style('whitegrid')
sns.countplot(data = repayment, x = 'paid_first_loan')
plt.xlabel('Loan Repayment')
plt.ylabel('Count')
plt.xticks([0, 1], ['Not Paid', 'Paid'])
plt.title('Distribution of Loan Repayment')
plt.show()


# In[47]:


# Histogram demonstrating the number of calls and its trend over the data. 
# Visually, we can see the counts with most frequency appears at the range between 0 to 1000 calls.
sns.histplot(data = new_repayment, x = 'calls_num', kde = True)


# In[48]:


# Boxplot with more clear view on four quartiles of the calls number. Median is a little above 2000, 
# and 50 % of the datafrom Q1 to Q3 ranges between 1500 to 4000 calls.
sns.boxplot(x=new_repayment['calls_num'])


# In[49]:


# Since the data records people with different lengths of time, checking the calls per day would be substantial. 
# The trending line shows a similiar pattern where the most frequency counts appear at 0 to 5 calls a day.
sns.histplot(data = new_repayment, x = 'phone_per_day', kde = True)


# In[50]:


# Specifically, according to the boxplot, the median of the calls is 5, and 75% of the calls falls under the range from 0 to 10.
sns.boxplot(data = new_repayment, x = 'phone_per_day')


# In[51]:


# Histograms showing the counts of number of calls per day for each user. Top one is those who paid, the second one is those did not.
# They have similiar trend line that the most frequent number of calls appear at 3 to 5 calls a day, and decreasing to the right-skewness.
fig, axes = plt.subplots(2, 1, figsize=(10,10))
ax1 = sns.histplot(data = new_repayment[new_repayment['paid_first_loan'] == 1], x = 'phone_per_day', kde = True, ax = axes[0])
ax1.set_xlabel('Number of Calls per Day')
ax1.set_title('Distribution of number of calls(Paid)')
ax2 = sns.histplot(data = new_repayment[new_repayment['paid_first_loan'] == 0], x = 'phone_per_day', kde = True, color = "red", ax = axes[1])
ax2.set_xlabel('Number of Calls per Day')
ax2.set_title('Distribution of number of calls(Unpaid)')


# In[52]:


# Combination of two plots above to compare.
sns.histplot(data = new_repayment[new_repayment['paid_first_loan'] == 1], x = 'phone_per_day', color = 'skyblue', kde = True)
sns.histplot(data = new_repayment[new_repayment['paid_first_loan'] == 0], x = 'phone_per_day', color = 'red', kde = True)
plt.xlabel('Number of Calls per Day')
plt.title('Distribution of number of calls')


# In[53]:


# Histograms showing the counts of average call duration for each user. Top one is those who paid, the second one is those did not.
# They have similiar trend line that the most frequent number of the length of calls appear at 30 to 50 mins, and decreasing to the right-skewness.
fig, axes = plt.subplots(2, 1, figsize=(10,10))
ax1 = sns.histplot(data = new_repayment[new_repayment['paid_first_loan'] == 1], x = 'AVG_DUR', kde = True, ax = axes[0])
ax1.set_xlabel('Average Call Duration')
ax1.set_title('Distribution average call duraion(Paid)')
ax2 = sns.histplot(data = new_repayment[new_repayment['paid_first_loan'] == 0], x = 'AVG_DUR', kde = True, color = "orange", ax = axes[1])
ax2.set_xlabel('Average Call Duration')
ax2.set_title('Distribution average call duraion(Unpaid)')


# In[54]:


# Box plot showing the counts of call duration for each user. Top one is those who paid, the second one is those did not.
# 
fig, axes = plt.subplots(2, 1, figsize=(10,10))
ax1 = sns.boxplot(data = new_repayment[new_repayment['paid_first_loan'] == 1], x = 'AVG_DUR', ax = axes[0])
ax1.set_xlabel('Average Call Duration')
ax1.set_title('Distribution average call duraion(Paid)')
ax1.set_xlim(0, 100)
ax2 = sns.boxplot(data = new_repayment[new_repayment['paid_first_loan'] == 0], x = 'AVG_DUR', color = "orange", ax = axes[1])
ax2.set_xlabel('Average Call Duration')
ax2.set_title('Distribution average call duraion(Unpaid)')
ax2.set_xlim(0,200)


# In[55]:


# Histograms showing the counts of max duration for each user. Top one is those who paid, the second one is those did not.
fig, axes = plt.subplots(2, 1, figsize=(10,10))
ax1 = sns.histplot(data = new_repayment[new_repayment['paid_first_loan'] == 1], x = 'MAX_DUR', kde = True, ax = axes[0])
ax1.set_xlabel('Max Call Duration')
ax1.set_title('Distribution the max call duraion(Paid)')
ax2 = sns.histplot(data = new_repayment[new_repayment['paid_first_loan'] == 0], x = 'MAX_DUR', kde = True, color = "orange", ax = axes[1])
ax2.set_xlabel('Max Call Duration')
ax2.set_title('Distribution the max call duraion(Unpaid)')


# In[56]:


# Box plot showing the counts of max duration for each user. Top one is those who paid, the second one is those did not.
# 
fig, axes = plt.subplots(2, 1, figsize=(10,10))
ax1 = sns.boxplot(data = new_repayment[new_repayment['paid_first_loan'] == 1], x = 'AVG_DUR', ax = axes[0])
ax1.set_xlabel('Max Call Duration')
ax1.set_title('Distribution the max call duraion(Paid)')
ax1.set_xlim(0,160)
ax2 = sns.boxplot(data = new_repayment[new_repayment['paid_first_loan'] == 0], x = 'AVG_DUR', color = "orange", ax = axes[1])
ax2.set_xlabel('Max Call Duration')
ax2.set_title('Distribution the max call duraion(Unpaid)')
ax2.set_xlim(0,160)


# In[57]:


# Histograms showing the counts of time length recorded for each user. Top one is those who paid, the second one is those did not.
fig, axes = plt.subplots(2, 1, figsize=(10,10))
ax1 = sns.histplot(data = new_repayment[new_repayment['paid_first_loan'] == 1], x = 'day_dif', kde = True, ax = axes[0])
ax1.set_xlabel('Time Length(days)')
ax1.set_title('Distribution time length(Paid)')
ax1.set_xlim(0,5000)
ax2 = sns.histplot(data = new_repayment[new_repayment['paid_first_loan'] == 0], x = 'day_dif', kde = True, color = "orange", ax = axes[1])
ax2.set_xlabel('Time length(days)')
ax2.set_title('Distribution time length(Unpaid)')
ax2.set_xlim(0,5000)


# In[58]:


# Box plot showing the time length recorded for each user. Top one is those who paid, the second one is those did not.
fig, axes = plt.subplots(2, 1, figsize=(10,10))
ax1 = sns.boxplot(data = new_repayment[new_repayment['paid_first_loan'] == 1], x = 'day_dif', ax = axes[0])
ax1.set_xlabel('Time Length(days)')
ax1.set_title('Distribution of time length(Paid)')
ax1.set_xlim(0,1000)
ax2 = sns.boxplot(data = new_repayment[new_repayment['paid_first_loan'] == 0], x = 'day_dif', color = "orange", ax = axes[1])
ax2.set_xlabel('Time Length(days)')
ax2.set_title('Distribution of time length(Unpaid)')
ax2.set_xlim(0,1000)


# In[59]:


merged_table['direction']


# In[60]:


# Percentage of all directions of calls
merged_table['direction'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.show()


# In[61]:


# Histograms showing the counts of the Incoming calls recorded for each user. Top one is those who paid, the second one is those did not.
fig, axes = plt.subplots(2, 1, figsize=(10,10))
ax1 = sns.histplot(data = new_repayment[new_repayment['paid_first_loan'] == 1], x = 'Incoming', kde = True, color = 'green', ax = axes[0])
ax1.set_xlabel('Incoming(Paid)')
ax1.set_title('Distribution of the Incoming number')
ax2 = sns.histplot(data = new_repayment[new_repayment['paid_first_loan'] == 0], x = 'Incoming', kde = True, color = "red", ax = axes[1])
ax2.set_xlabel('Incoming(Unpaid)')
ax2.set_title('Distribution of the Incoming number')


# In[62]:


# Histograms showing the counts of time length recorded for each user. Top one is those who paid, the second one is those did not.
fig, axes = plt.subplots(2, 1, figsize=(10,10))
ax1 = sns.histplot(data = new_repayment[new_repayment['paid_first_loan'] == 1], x = 'Outgoing', kde = True, color = 'green', ax = axes[0])
ax1.set_xlabel('Outcoming(Paid)')
ax1.set_title('Distribution of the Outgoing number')
ax2 = sns.histplot(data = new_repayment[new_repayment['paid_first_loan'] == 0], x = 'Outgoing', kde = True, color = "red", ax = axes[1])
ax2.set_xlabel('Outcoming(Unpaid)')
ax2.set_title('Distribution of the Outgoing number')


# In[63]:


# Histograms showing the counts of time length recorded for each user. Top one is those who paid, the second one is those did not.
fig, axes = plt.subplots(2, 1, figsize=(10,10))
ax1 = sns.histplot(data = new_repayment[new_repayment['paid_first_loan'] == 1], x = 'Weekday_num', kde = True, ax = axes[0])
ax1.set_xlabel('Weekday umbers(Paid)')
ax1.set_title('Distribution of the Weekday number')
ax2 = sns.histplot(data = new_repayment[new_repayment['paid_first_loan'] == 0], x = 'Weekday_num', kde = True, color = "orange", ax = axes[1])
ax2.set_xlabel('Weekday number(Unpaid)')
ax2.set_title('Distribution of the Weekday number')


# In[64]:


# Box plot showing the time length recorded for each user. Top one is those who paid, the second one is those did not.
fig, axes = plt.subplots(2, 1, figsize=(10,10))
ax1 = sns.boxplot(data = new_repayment[new_repayment['paid_first_loan'] == 1], x = 'Weekday_num', ax = axes[0])
ax1.set_xlabel('Weekday number(Paid)')
ax1.set_title('Distribution of the weekday number(Paid)')
ax2 = sns.boxplot(data = new_repayment[new_repayment['paid_first_loan'] == 0], x = 'Weekday_num', color = "orange", ax = axes[1])
ax2.set_xlabel('Weekday number(unpaid)')
ax2.set_title('Distribution of the weekday number(unPaid)')


# In[65]:


# Histograms showing the counts of time length recorded for each user. Top one is those who paid, the second one is those did not.
fig, axes = plt.subplots(2, 1, figsize=(10,10))
ax1 = sns.histplot(data = new_repayment[new_repayment['paid_first_loan'] == 1], x = 'Weekend_num', kde = True, color = 'green', ax = axes[0])
ax1.set_xlabel('Weekend numbers(Paid)')
ax1.set_title('Distribution of the Weekend number')
ax2 = sns.histplot(data = new_repayment[new_repayment['paid_first_loan'] == 0], x = 'Weekend_num', kde = True, color = "red", ax = axes[1])
ax2.set_xlabel('Weekend number(Unpaid)')
ax2.set_title('Distribution of the Weekend number')


# In[66]:


# Box plot showing the time length recorded for each user. Top one is those who paid, the second one is those did not.
fig, axes = plt.subplots(2, 1, figsize=(10,10))
ax1 = sns.boxplot(data = new_repayment[new_repayment['paid_first_loan'] == 1], x = 'Weekend_num', color = 'green', ax = axes[0])
ax1.set_xlabel('Weekend number(Paid)')
ax1.set_title('Distribution of the weekend number(Paid)')
ax2 = sns.boxplot(data = new_repayment[new_repayment['paid_first_loan'] == 0], x = 'Weekend_num', color = "red", ax = axes[1])
ax2.set_xlabel('Weekend number(unpaid)')
ax2.set_title('Distribution of the weekend number(unPaid)')


# In[67]:


# Histograms showing the counts of time length recorded for each user. Top one is those who paid, the second one is those did not.
fig, axes = plt.subplots(4, 1, figsize=(10,20))
ax1 = sns.histplot(data = new_repayment[new_repayment['paid_first_loan'] == 1], x = 'Morning_ratio', kde = True, color = 'blue', ax = axes[0])
ax1.set_xlabel('Morning Ratio(Paid)')
ax1.set_title('Distribution of the Morning Ratio')
ax2 = sns.histplot(data = new_repayment[new_repayment['paid_first_loan'] == 0], x = 'Morning_ratio', kde = True, color = "orange", ax = axes[1])
ax2.set_xlabel('Morning Ratio(Unpaid)')
ax2.set_title('Distribution of the Morning Ratio')

ax3 = sns.histplot(data = new_repayment[new_repayment['paid_first_loan'] == 1], x = 'Afternoon_ratio', kde = True, color = 'green', ax = axes[2])
ax3.set_xlabel('Afternoon Ratio(Paid)')
ax3.set_title('Distribution of the Afternoon Ratio')
ax4 = sns.histplot(data = new_repayment[new_repayment['paid_first_loan'] == 0], x = 'Afternoon_ratio', kde = True, color = "red", ax = axes[3])
ax4.set_xlabel('Afternoon Ratio(Unpaid)')
ax4.set_title('Distribution of the Afternoon Ratio')


# In[68]:


fig, axes = plt.subplots(4, 1, figsize=(10,20))
ax1 = sns.histplot(data = new_repayment[new_repayment['paid_first_loan'] == 1], x = 'Evening_ratio', kde = True, color = 'blue', ax = axes[0])
ax1.set_xlabel('Evening Ratio(Paid)')
ax1.set_title('Distribution of the Evening Ratio')
ax2 = sns.histplot(data = new_repayment[new_repayment['paid_first_loan'] == 0], x = 'Evening_ratio', kde = True, color = "orange", ax = axes[1])
ax2.set_xlabel('Evening Ratio(Unpaid)')
ax2.set_title('Distribution of the Evening Ratio')

ax3 = sns.histplot(data = new_repayment[new_repayment['paid_first_loan'] == 1], x = 'Night_ratio', kde = True, color = 'green', ax = axes[2])
ax3.set_xlabel('Night Ratio(Paid)')
ax3.set_title('Distribution of the Night Ratio')
ax4 = sns.histplot(data = new_repayment[new_repayment['paid_first_loan'] == 0], x = 'Night_ratio', kde = True, color = "red", ax = axes[3])
ax4.set_xlabel('Night Ratio(Unpaid)')
ax4.set_title('Distribution of the Night Ratio')


# In[69]:


# From this histoplot, assuming people who has over 3.75% of the a user's call are marked as friend, over 7.5% are marked as close friend, 
# others are marked as known 
sns.histplot(data = calls, x = 'caller_percentage', kde = True),
plt.ylim(0, 30000)
plt.xlim(0,0.6)


# In[70]:


# Histograms showing the counts of time length recorded for each user. Top one is those who paid, the second one is those did not.
fig, axes = plt.subplots(6, 1, figsize=(10,30))
ax1 = sns.histplot(data = new_repayment[new_repayment['paid_first_loan'] == 1], x = 'Normal', kde = True, color = 'blue', ax = axes[0])
ax1.set_xlabel('Normal Ratio(Paid)')
ax1.set_title('Distribution of the Normal Ratio')
ax2 = sns.histplot(data = new_repayment[new_repayment['paid_first_loan'] == 0], x = 'Normal', kde = True, color = "orange", ax = axes[1])
ax2.set_xlabel('Normal Ratio(Unpaid)')
ax2.set_title('Distribution of the Normal Ratio')

ax3 = sns.histplot(data = new_repayment[new_repayment['paid_first_loan'] == 1], x = 'Close', kde = True, color = 'blue', ax = axes[2])
ax3.set_xlabel('Close Ratio(Paid)')
ax3.set_title('Distribution of the Close Ratio')
ax4 = sns.histplot(data = new_repayment[new_repayment['paid_first_loan'] == 0], x = 'Close', kde = True, color = "orange", ax = axes[3])
ax4.set_xlabel('Close Ratio(Unpaid)')
ax4.set_title('Distribution of the Close Ratio')

ax3 = sns.histplot(data = new_repayment[new_repayment['paid_first_loan'] == 1], x = 'Very Close', kde = True, color = 'blue', ax = axes[4])
ax3.set_xlabel('Very Close Ratio(Paid)')
ax3.set_title('Distribution of the Very Close Ratio')
ax4 = sns.histplot(data = new_repayment[new_repayment['paid_first_loan'] == 0], x = 'Very Close', kde = True, color = "orange", ax = axes[5])
ax4.set_xlabel('Very Close Ratio(Unpaid)')
ax4.set_title('Distribution of the Very Close Ratio')


# In[71]:


plt.figure(figsize = (20,20))
sns.heatmap(new_repayment.corr(), annot=True, cmap='coolwarm')


# # Feature Engineering

# In[72]:


correlations = new_repayment.corr()['paid_first_loan'].sort_values()
correlations


# In[73]:


new_repayment = new_repayment.fillna(0)
new_repayment.isnull().sum()


# In[74]:



# new_repayment['signup_year'] = new_repayment['signup_date'].dt.year
# new_repayment['signup_month'] = new_repayment['signup_date'].dt.month
# new_repayment['signup_day'] = new_repayment['signup_date'].dt.day

# new_repayment['disbursement_date'] = pd.to_datetime(new_repayment['disbursement_date'])
# new_repayment['disbursement_year'] = new_repayment['disbursement_date'].dt.year
# new_repayment['disbursement_month'] = new_repayment['disbursement_date'].dt.month
# new_repayment['disbursement_day'] = new_repayment['disbursement_date'].dt.day

new_repayment = new_repayment.replace([np.inf, -np.inf], 0)
features = new_repayment.drop(columns=['paid_first_loan', 'signup_date', 'disbursement_date'])
target = new_repayment['paid_first_loan']

# Initialize the model
model = RandomForestClassifier(random_state=0)

# Fit the model
model.fit(features, target)

# Get feature importances
importances = model.feature_importances_

# Let's print each feature with its  importance
for feature_name, importance in zip(features.columns, importances):
    print(f"Feature: {feature_name}, Importance: {importance}")


# In[75]:


merged_table.head()


# In[76]:


feature_2 = merged_table.drop(columns = ['timestamp', 'phone', 'name', 'caller_id', 'time', 'signup_date', 'disbursement_date', 'paid_first_loan'])
feature_2['direction'] = feature_2['direction'].astype(int)
feature_2['Relation'] = feature_2['Relation'].apply(lambda x:0 if x=='Normal' else 1 if x=='Close' else 2)
feature_2['day_type'] = feature_2['day_type'].replace({'Weekday':0, 'Weekend':1})
feature_2['part_of_day'] = feature_2['part_of_day'].apply(lambda x:0 if x=='Morning' else 1 if x=='Afternoon' else 2 if x == 'Evening' else 3)

target2 = merged_table['paid_first_loan']
feature_2 = feature_2.fillna(0)
feature_2 = feature_2.replace([np.inf, -np.inf], 0)
# Initialize the model
model2 = RandomForestClassifier(random_state=0)

# Fit the model
model2.fit(feature_2, target2)

# Get feature importances
importances2 = model2.feature_importances_

# Let's print each feature with its  importance
for feature_name, importance in zip(feature_2.columns, importances):
    print(f"Feature: {feature_name}, Importance: {importance}")


# In[81]:


# Create feature matrix (X) and target vector (y)
X = new_repayment.drop(columns=['paid_first_loan', 'signup_date', 'disbursement_date'])
y = new_repayment['paid_first_loan']

# Store the column names of X before transformation
X_columns = X.columns

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

# Instantiate the Logistic Regression model
model = LogisticRegression()

# Fit the model
model.fit(X_train, y_train)

# Get the feature importance
importance = np.abs(model.coef_[0])

# Print the feature importance
for i, value in enumerate(importance):
    print(f"Feature {X_columns[i]}: {value}")

