# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Silly formatting stuff
rcParams.update({'figure.autolayout': True})
pd.set_option('display.width', 320)
pd.set_option("display.max_columns", 12)

# Read in the data
titanic_train = pd.read_csv("C:/Users/joshs/PycharmProjects/incoherent-rabble/Kaggle/Titanic/Data/train.csv")
titanic_test = pd.read_csv("C:/Users/joshs/PycharmProjects/incoherent-rabble/Kaggle/Titanic/Data/test.csv")
print(titanic_train.head())

# Some summary statistics
print(titanic_train.describe())
print(titanic_train.info())

print(titanic_train.groupby(["Sex","Survived"]).count())
print(titanic_train.groupby(["Embarked", "Survived"]).count())

# Pair-plot
no_nas = titanic_train.dropna()
sns.pairplot(no_nas, hue = "Sex")
#plt.show()

# Correlation plot
corr_mat = titanic_train.corr()
cmap = sns.diverging_palette(10, 220, as_cmap=True)
sns.heatmap(corr_mat, vmin=-1, vmax=1, cmap=cmap, cbar_kws={"shrink": 0.75}, annot=True)
#plt.show()

# Begin model building
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier


titanic_train.drop(labels=['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
titanic_train = pd.get_dummies(titanic_train)

titanic_test.drop(labels=['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
titanic_test = pd.get_dummies(titanic_test)

steps = [('imputation', Imputer(strategy='mean', axis=1, verbose=1)),
         ('clf', RandomForestClassifier(verbose=1))]

pipeline = Pipeline(steps)

model = pipeline.fit(titanic_train.drop('Survived', axis=1), y=titanic_train.Survived)

print(pipeline.steps[1][1].feature_importances_)

predicted = pd.DataFrame(model.predict(titanic_test),
                         columns=['Survived'])

predicted['PassengerId'] = titanic_test['PassengerId']

predicted.to_csv('C:/Users/joshs/PycharmProjects/incoherent-rabble/Kaggle/Titanic/Data/predicted.csv', index=False)
