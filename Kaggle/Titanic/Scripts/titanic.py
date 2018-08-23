# Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Silly formatting stuff
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
pd.set_option('display.width', 320)
pd.set_option("display.max_columns", 12)

# Read in the data
titanic_raw = pd.read_csv("C:/Users/joshs/PycharmProjects/incoherent-rabble/Kaggle/Titanic/Data/train.csv")
print(titanic_raw.head())

# Some summary statistics
print(titanic_raw.describe())
print(titanic_raw.info())

print(titanic_raw.groupby(["Sex","Survived"]).count())
print(titanic_raw.groupby(["Embarked", "Survived"]).count())

# Pair-plot
no_nas = titanic_raw.dropna()
sns.pairplot(no_nas, hue = "Sex")
plt.show()

# Correlation plot
corr_mat = titanic_raw.corr()
cmap = sns.diverging_palette(10, 220, as_cmap=True)
sns.heatmap(corr_mat, vmin=-1, vmax=1, cmap=cmap, cbar_kws={"shrink": 0.75})
plt.show()