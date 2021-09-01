import pandas
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, plot_roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier



def randomForest(df):
    header = df.columns
    X = df.drop(header[-1], axis=1)
    y = df.drop(header[:-1], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.20)
    random_forest = RandomForestClassifier(criterion='entropy')
    random_forest.fit(X_train,y_train.values.ravel())
    y_hat = random_forest.predict(X=X_test)
    matrix = confusion_matrix(y_true=y_test,y_pred=y_hat)
    soundness = (matrix[0][0] + matrix[1][1]) / (sum(matrix)[0] + sum(matrix)[1])
    print("Random Forest results")
    print(soundness)
    roc = plot_roc_curve(random_forest,X_test,y_test)
    
    
def decisionTree(df):
    header = df.columns
    X = df.drop(header[-1], axis=1)
    y = df.drop(header[:-1], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.20)
    decision_tree = DecisionTreeClassifier(criterion='entropy')
    decision_tree.fit(X_train,y_train.values.ravel())
    y_hat = decision_tree.predict(X_test)
    matrix = confusion_matrix(y_true=y_test,y_pred=y_hat)
    soundness = (matrix[0][0] + matrix[1][1]) / (sum(matrix)[0] + sum(matrix)[1])
    print("Decision Tree results")
    print(soundness)
    roc = plot_roc_curve(decision_tree,X_test,y_test)


df = pandas.read_csv('diabetes.csv')
randomForest(df)
decisionTree(df)
plt.show()
