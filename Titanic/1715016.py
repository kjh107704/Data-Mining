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
# define initial value of w
initial_value_of_w = 0.1
W = np.array([initial_value_of_w for _ in range(num)], dtype='f')
X = np.array()


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

# main function
def main():
    dataPreprocessing()


main()
