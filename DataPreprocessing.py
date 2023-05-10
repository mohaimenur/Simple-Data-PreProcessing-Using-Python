import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data_set=pd.read_csv("Data.csv")
x=data_set.iloc[:,:-1].values
y=data_set.iloc[:,3].values

imp=SimpleImputer(missing_values=np.nan, strategy="mean")
Imputer=imp.fit(x[:,1:3])
x[:,1:3]=imp.fit_transform(x[:,1:3])

#label_encoder_x= LabelEncoder()
#x[:,0]=label_encoder_x.fit_transform(x[:,0])

one_hot_encoder_x=ColumnTransformer([("Country",OneHotEncoder(),[0])], remainder="passthrough")
x=one_hot_encoder_x.fit_transform(x)

label_encoder_y= LabelEncoder()
y=label_encoder_y.fit_transform(y)

x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.2, random_state=0)


st_x=StandardScaler()
x_train=st_x.fit_transform(x_train)
x_test=st_x.fit_transform(x_test)