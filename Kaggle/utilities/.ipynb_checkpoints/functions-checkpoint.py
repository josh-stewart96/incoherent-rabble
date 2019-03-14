import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# A simple function that plots the percentage of NA values in each column in a horizontal bar plot
def plotNAs(data, figsize=(15,15), color='red'):
    
    try:
        num_nulls = data.isna().sum()
        nulls_pct = pd.DataFrame({'NA_pct': [(x / 1460) * 100 for x in num_nulls], 'colnames': data.columns.values})
        nulls_pct.sort_values(by='NA_pct', inplace=True)
    except TypeError:
        print("TypeError: Input must be a pandas DataFrame.")
        return
    
    ax, fig = plt.subplots(figsize=figsize)
    sns.barplot(data=nulls_pct, x='NA_pct', y='colnames', color='red')
    plt.title("Plot of the percentage of NA values in each column")
    plt.xlabel("NA Percentage")
    plt.ylabel("Feature")


    



