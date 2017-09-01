import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing dataset
dataset = pd.read_csv('Data.csv')
#Takes all values except last row
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values


#Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
#Fitting the missing values into dataset X
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

#Encoding Categorial data
#-----Countries-----
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#label countries by either 0,1,2 (has order)
labelencoder_X = LabelEncoder()
x[:, 0] = labelencoder_X.fit_transform(x[:, 0])
#using oneHotEncoder by bits for unordered list
onehotencoder = OneHotEncoder(categorical_features = [0])
x = onehotencoder.fit_transform(x).toarray()

#-----Dependent Variable(yes/no)-----
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#Splitting dataset into training and test
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#Feature scaling: Makes sure euclidean distance isn't dominated by 1 set of values
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)






