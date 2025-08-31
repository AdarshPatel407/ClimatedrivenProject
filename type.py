import pandas as pd
import numpy as np
import matplotlib as mplt

df=pd.read_csv("update_temperature.csv",encoding="latin1")


print(df)

print(df.info())
#to understand max and min loss
print(df.describe())

#Finding missing values
print(df.isnull().sum())