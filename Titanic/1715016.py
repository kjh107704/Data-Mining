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

    for i, index in enumerate(feature_index):
        plt.subplot(1, feature_size + 1, i + 1, aspect='equal')
        plt.pie([survived[index], dead[index]], labels=[
                'Survived', 'Dead'], autopct='%1.1f%%')
        plt.title(str(index) + '\'s ratio')

    plt.show()
'''
# pie_chart('Embarked')


def find_NaN(feature):
    feature_ratio = train[feature].value_counts(sort=False)
    feature_size = feature_ratio.size
    feature_index = feature_ratio.index

    print("feature_index = "+feature_index)


col = train.columns
print(col[1])
