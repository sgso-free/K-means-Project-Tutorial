#pip install -r ../requirements.txt

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt  
import pickle

from sklearn.preprocessing import StandardScaler 
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#Import the data to dataframe
url='https://raw.githubusercontent.com/4GeeksAcademy/k-means-project-tutorial/main/housing.csv'
df_raw=pd.read_csv(url)

#select only thhis 3 column 
df=df_raw[['MedInc','Latitude','Longitude']]

print(df)

#Scaler the data
stScaler = StandardScaler()
df_norm = stScaler.fit_transform(df) 

print(df_norm)

#use 2 cluster, the best after analize using siluete and elbow methods
kmeans = KMeans(n_clusters=2, random_state=100)
kmeans.fit(df_norm)
 
#df2 = stScaler.inverse_transform(df_norm)
df2=pd.DataFrame(df_norm,columns=['MedInc','Latitude','Longitude'])
df2['Cluster'] = kmeans.labels_
print(df2[df2['Cluster']==0]) 
print(df2[df2['Cluster']==1]) 

#save the model to file
filename = 'models/finalized_model.sav' #use absolute path
pickle.dump(kmeans, open(filename, 'wb'))
 
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))

# get the predict : means which cluster it belongs to
print('Predicted ] : \n', loaded_model.predict([[ 2.344766,  1.052548,  -1.327835], [ -0.044727,0.542225, 0.329279]])) 
