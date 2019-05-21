import seaborn as sns
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.show()
sns.set()

# read csv files
train = pd.read_csv(sys.argv[1])
test = pd.read_csv(sys.argv[2])

# data preprocessing
def dataPreprocessing():
    print("print train.shape")
    print(train.shape)
    train_columns = train.columns
    print("print data columns")
    print(train_columns)
    print("number of columns")
    print(train_columns.size)
    print(train[train_columns[2]].head())


    for i in range(train_columns.size):
        print("\n" + train_columns[i] + "\'s Data")
        print(train[train_columns[i]].unique())
        print("\n"+ train_columns[i] + "\'s NaN")
        print("before dropna: ")
        print(train[train_columns[i]].size)
        
        print("after dropna: ")
        print(train[train_columns[i]].dropna().size)


# define number of attributes
num = 1
'''
def pie_chart(feature):
    feature_ratio = train[feature].value_counts(sort=False)
    feature_size = feature_ratio.size
    feature_index = feature_ratio.index
    survived = train[train['Survived'] == 1][feature].value_counts()
    dead = train[train['Survived'] == 0][feature].value_counts()

    plt.plot(aspect='auto')
    plt.pie(feature_ratio, labels=feature_index, autopct='%1.1f%%')
    plt.title(feature + '\'s ratio in total')
    plt.show()

# z function
def z():
    return np.sum(np.multiply(W, X))

# Logistic Regression Function
def Logistic_Regression():
    if (1 / 1 + np.exp(-z()) >= 0.5):
        return 1
    else:
        return 0
'''
# pie_chart('Embarked')


def find_NaN(feature):
    feature_ratio = train[feature].value_counts(sort=False)
    feature_size = feature_ratio.size
    feature_index = feature_ratio.index

# main function
def main():
    dataPreprocessing()


main()
