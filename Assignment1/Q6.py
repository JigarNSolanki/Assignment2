import pandas as pd
df = pd.read_csv("banking.csv")

#--------- Unique Values in Education-----------
print(list(df['education'].unique()))

#-------Number of Customer Subscribed or not---------
print(df['y'].value_counts())

#---- mean values of independent variables for every y-----
print(df[df.y == 1].mean())
print(df[df.y == 0].mean())

#------------- mean age for every marital status-----------
print(df[df.marital == 'married'].age.mean())
print(df[df.marital == 'single'].age.mean())
print(df[df.marital == 'divorced'].age.mean())
print(df[df.marital == 'unknown'].age.mean())

#-----Check Null values-------------
df.isna()