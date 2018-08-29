# Import libraries
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import rcParams

# Silly formatting stuff
rcParams.update({'figure.autolayout': True})
pd.set_option('display.width', 320)
pd.set_option("display.max_columns", 12)

# Read in the data
titanic_train = pd.read_csv("C:/Users/joshs/PycharmProjects/incoherent-rabble/Kaggle/Titanic/Data/train.csv")
titanic_test = pd.read_csv("C:/Users/joshs/PycharmProjects/incoherent-rabble/Kaggle/Titanic/Data/test.csv")
# print(titanic_train.head())

# Engineering
# Extract first letter from cabin number
titanic_train.Cabin = titanic_train.Cabin.str[0]
titanic_test.Cabin = titanic_test.Cabin.str[0]

# Extract title from name
titanic_train['Title'] = titanic_train.Name.str.split(', ', expand=True)[1].str.split('.', expand=True)[0]
titanic_test['Title'] = titanic_test.Name.str.split(', ', expand=True)[1].str.split('.', expand=True)[0]

# Begin model building
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

# Remove unnecessary columns
titanic_train.drop(labels=['Name', 'Ticket'], axis=1, inplace=True)
titanic_test.drop(labels=['Name', 'Ticket'], axis=1, inplace=True)

# Ensure columns are the same for train and test when one-hot-encoding
train_ind = len(titanic_train)
combined = pd.concat(objs=[titanic_train,titanic_test], axis=0, sort=False)
combined = pd.get_dummies(combined)
titanic_train = combined[:train_ind]
titanic_test = combined[train_ind:].drop(['Survived'], axis=1)


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(titanic_train.drop(['Survived'], axis=1),
                                                    titanic_train.Survived,
                                                    random_state=np.random.seed(42))

# # Number of trees in random forest
# n_estimators = [int(x) for x in np.linspace(start=50, stop=150, num=5)]
# # Number of features to consider at every split
# max_features = ['auto', 'sqrt']
# # Maximum number of levels in tree
# max_depth = [int(x) for x in np.linspace(10, 110, num=5)]
# max_depth.append(None)
# # Minimum number of samples required to split a node
# min_samples_split = [2, 5, 10]
# # Minimum number of samples required at each leaf node
# min_samples_leaf = [1, 2, 4]
# # Method of selecting samples for training each tree
# bootstrap = [True, False]
# Create the random grid
# random_grid = {'rf__n_estimators': n_estimators,
#                'rf__max_features': max_features,
#                'rf__max_depth': max_depth,
#                'rf__min_samples_split': min_samples_split,
#                'rf__min_samples_leaf': min_samples_leaf,
#                'rf__bootstrap': bootstrap}


# Initialise pipeline
steps = [('imputation', Imputer(strategy='mean', axis=1)),
         ('rf', RandomForestClassifier(n_estimators=125,
                                       min_samples_split=10,
                                       min_samples_leaf=4,
                                       max_features='sqrt',
                                       max_depth=35,
                                       bootstrap=True,
                                       random_state=np.random.seed(123)
                                       ))]

pipeline = Pipeline(steps)

#rf_random = RandomizedSearchCV(estimator=pipeline, param_distributions=random_grid, n_iter=100, cv=10, verbose=2, random_state=42)

# Fit pipeline
model = pipeline.fit(titanic_train.drop(['Survived'], axis=1), titanic_train.Survived)

# Predict new values
predicted = pd.DataFrame(model.predict(titanic_test),
                         columns=['Survived'])

# Confusion matrix
#print(pd.DataFrame(confusion_matrix(y_test, predicted),columns=['0', '1']))

# ROC-AUC Score
#print("AUC score: {}".format(roc_auc_score(y_test, predicted)))

predicted['PassengerId'] = titanic_test['PassengerId']
predicted['Survived'] = predicted['Survived'].astype('Int64')

predicted.to_csv('C:/Users/joshs/PycharmProjects/incoherent-rabble/Kaggle/Titanic/Data/predicted.csv', index=False)
