import os
import numpy as np
import pandas as pd


# feature set paths
SNAPSHOT_1 = 'outputs/snapshot_1.csv'
SNAPSHOT_2 = 'outputs/snapshot_2.csv'
SNAPSHOT_3 = 'outputs/snapshot_3.csv'
SNAPSHOT_4 = 'outputs/snapshot_4.csv'
SNAPSHOT_5 = 'outputs/snapshot_5.csv'

# get the feature sets
df1=pd.read_csv(SNAPSHOT_1)
df2=pd.read_csv(SNAPSHOT_2) 
df3=pd.read_csv(SNAPSHOT_3)
df4=pd.read_csv(SNAPSHOT_4)
df5=pd.read_csv(SNAPSHOT_5) 


# sort a dataframe based on an attribute 
def sort_df_names(df, attribute='filename'):
    train_size = int(len(df) * 0.8)
    
    train_df = df[:train_size].copy()
    test_df = df[train_size:1+len(df)].copy()

    train_df = train_df.sort_values(attribute,ascending=True)
    test_df = test_df.sort_values(attribute,ascending=True)

    df = pd.concat([train_df, test_df], axis=0)
    return df


# to concat feature sets obtained from each snapshot
def concat_features():
	# making variables global so as to access them
	global df1, df2, df3, df4, df5

	labels = df1['label'].copy()
	filenames = df1['filename'].copy()

	df1=df1.drop(['label','filename'],axis=1)
	df2=df2.drop(['label','filename'],axis=1)
	df3=df3.drop(['label','filename'],axis=1)
	df4=df4.drop(['label','filename'],axis=1)
	df5=df5.drop(['label','filename'],axis=1)

	df_concat = pd.concat([df1,df2,df3,df4,df5], axis=1)
	df_concat['label'] = labels
	df_concat['filename'] = filenames

	return df_concat


# function that prepares the concatenated feature set for FS
def get_feature_set():
	# making variables global so as to access them
	global df1, df2, df3, df4, df5

	# sort the feature sets w.r.t. filenames for concatenation
	df1 = sort_df_names(df1)
	df2 = sort_df_names(df2)
	df3 = sort_df_names(df3)
	df4 = sort_df_names(df4)
	df5 = sort_df_names(df5)

	# finally, concatenate the feature sets
	df_concat = concat_features()
	return df_concat

