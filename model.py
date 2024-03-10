import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


dataset = pd.read_csv('./Caesarian Section Classification Dataset(CSV).csv')

dataset = dataset.rename(columns={'Delivery No': 'Delivery time'})




lb = LabelEncoder()
dataset['Delivery time'] = lb.fit_transform(dataset['Delivery time'])
dataset['Blood of Pressure'] = lb.fit_transform(dataset['Blood of Pressure'])
dataset['Heart Problem'] = lb.fit_transform(dataset['Heart Problem'])
dataset['Caesarian'] = lb.fit_transform(dataset['Caesarian'])



X = dataset.iloc[:, 0:5].values
y = dataset.iloc[:, 5].values


X_train, X_test, y_train, y_test =train_test_split(X,y,test_size= 0.25, random_state=0)


model = SVC(kernel='linear', C=1.0, random_state=None)


model.fit(X_train, y_train)

pickle.dump(model, open("model.pkl","wb"))

