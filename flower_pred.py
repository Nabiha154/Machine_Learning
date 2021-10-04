import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris()
df = pd.DataFrame(iris.data, columns = iris.feature_names)
df["target"] = iris.target
df["flower_names"] = df.target.apply(lambda x: iris.target_names[x])
import matplotlib.pyplot as plt
%matplotlib inline
df_first = df[df.target==0]
df_sec = df[df.target==1]
df_third = df[df.target==2]
plt.scatter(df_first['sepal length (cm)'],df_first['sepal width (cm)'], color = 'red',marker='+')
plt.scatter(df_sec['sepal length (cm)'],df_sec['sepal width (cm)'], color = 'green',marker='+')
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
#plt.scatter(df_third['sepal length (cm)'],df_third['sepal width (cm)'], color = 'blue',marker='+')
plt.scatter(df_first['petal length (cm)'],df_first['petal width (cm)'], color = 'red',marker='+')
plt.scatter(df_sec['petal length (cm)'],df_sec['petal width (cm)'], color = 'green',marker='+')
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
#plt.scatter(df_third['sepal length (cm)'],df_third['sepal width (cm)'], color = 'blue',marker='+')
from sklearn.model_selection import train_test_split
X = df.drop(['target','flower_names'], axis='columns')
Y = df.target
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)
from sklearn.svm import SVC 
model = SVC()
model.fit(X_train,Y_train)
#Sepal length: 5.84 cm. Sepal width: 3.05 cm. Petal length: 3.76 cm. Petal width: 1.20 cm. Correct answer is versicolor
model.predict([[5.84,3.05,3.76,1.2]])
y_pred = model.predict(X_test)
from sklearn.metrics import confusion_matrix
cm =confusion_matrix(Y_test , y_pred)
import seaborn as sns
plt.figure(figsize=(10,7))
sns.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
#model score is 1.00
