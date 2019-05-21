import seaborn as sns
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
plt.show()
sns.set()

# read csv files
train = pd.read_csv(sys.argv[1])
test = pd.read_csv(sys.argv[2])
data = train


def add_age(cols):  # impute average age values to null age values
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        return int(data[data["Pclass"] == Pclass]["Age"].mean())
    else:
        return Age


def add_embarked(cols):  # impute S to null Embarked values
    Embarked = cols[0]
    if pd.isnull(Embarked):
        return str('S')
    else:
        return Embarked


def MappingEmbarked(cols):  # mapping Embarked values to number
    Embarked = cols[0]
    if Embarked == 'S':
        return 3
    elif Embarked == 'C':
        return 2
    else:
        return 1


def AppendFamily(cols):  # calculate and append Family = SibSp + Parch
    SibSp = cols[0]
    Parch = cols[1]

    if pd.isnull(SibSp):
        if pd.isnull(Parch):
            return 0
        else:
            return Parch
    elif pd.isnull(Parch):
        return SibSp
    else:
        return Parch+SibSp


def LogFare(cols):  # log to Fare
    Fare = cols[0]
    if Fare > 0:
        return np.log(Fare)
    else:
        return 0


def dataPreprocessing(dataSet):  # dataPreprocessing

    # preprocessing Age
    # impute average age values to null age values
    dataSet["Age"] = dataSet[["Age", "Pclass"]].apply(add_age, axis=1)

    # preprocessing Embarked
    # impute S to null Embarked values
    dataSet["Embarked"] = dataSet[["Embarked"]].apply(add_embarked, axis=1)
    # mapping Embarked values to number
    dataSet["Embarked"] = dataSet[["Embarked"]].apply(MappingEmbarked, axis=1)

    # preprocessing SibSp, Parch
    # calculate and append Family = SibSp + Parch
    dataSet["Family"] = dataSet[["SibSp", "Parch"]].apply(AppendFamily, axis=1)

    # preprocessing Fare
    # log to Fare
    dataSet["Fare"] = dataSet[["Fare"]].apply(LogFare, axis=1)

    # preprocessing Sex
    pd.get_dummies(dataSet["Sex"])
    sex = pd.get_dummies(dataSet["Sex"], drop_first=True)
    dataSet = pd.concat([dataSet, sex], axis=1)

    # drop attribute
    dataSet.drop(["Cabin", "PassengerId",  "Name", "Sex", "Ticket",
                  "SibSp", "Parch"], axis=1, inplace=True)
    return dataSet


def z(W, X):  # return z
    z = np.dot(X, W[1:])+W[0]
    return 1.0 / (1.0 + np.exp(-z))


def CostFunction(W, X, y, m, a, cnt):  # update W for cnt times
    for i in range(cnt):
        hx = z(W, X)
        error = hx - y
        grad = X.T.dot(error)
        grad = grad*((1.0 / m))
        W[0] = W[0] - a * error.sum()
        W[1:] = W[1:] - a * grad
        print("iteration = "+str(i))
        print(W)
    return W


def MappingResult(cols):  # mapping z value to binary
    _z = cols[0]
    if _z >= 0.5:
        return 1
    else:
        return 0


def Logistic_Regression(W, test):  # Logistic Regression Function
    result = z(W, test)
    return result


def main():  # main function

    # define number of attributes
    num = 6

    # define initial value of w
    initial_value_of_w = 0
    W = np.array([initial_value_of_w for _ in range(num+1)], dtype='f')

    # dataPreprocessing
    global train
    train = dataPreprocessing(train)

    # make train X, y data
    train_X = train.drop("Survived", axis=1)
    train_y = train["Survived"]
    W = CostFunction(W, train_X, train_y, train_y.size, 0.001, 100000)

    global test
    global data

    result_data = pd.DataFrame(data=test['PassengerId'])
    result_data = result_data.set_index("PassengerId")
    data = test
    test = dataPreprocessing(test)
    print(result_data.shape)
    result = Logistic_Regression(W, test)
    result_data["Survived"] = result
    test = test.drop("male", axis=1)
    result_data["Survived"] = result_data[[
        "Survived"]].apply(MappingResult, axis=1)
    result_data.to_csv("1715016.csv", mode='w')


main()
