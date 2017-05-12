from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer

#loads the dataset
data = load_breast_cancer()

#Splits dataset into training and test set
x_train, x_test, y_train, y_test = train_test_split(data['data'],data['target'], random_state=0)

#Creates a gradient boosting regression tree with default parameters
gb = GradientBoostingClassifier(random_state=0)
gb.fit(x_train, y_train)

print("The training set accuracy is " + str(gb.score(x_train,y_train)) )
print("The training set accuracy is " + str(gb.score(x_test,y_test)) )

#Creates a gradient boosting regression tree with max depth 1
#max depth usually set lower than 5
gb2 = GradientBoostingClassifier(random_state=0, max_depth=1)
gb2.fit(x_train, y_train)

print("The training set accuracy, with depth 1, is " + str(gb2.score(x_train,y_train)) )
print("The training set accuracy, with depth 1, is " + str(gb2.score(x_test,y_test)) )

#Creates a gradient boosting regression tree with learning rate 0.01
#learning rate is the degree of which the next tree corrects the mistake of the previous tree
gb3 = GradientBoostingClassifier(random_state=0, learning_rate=0.01)
gb3.fit(x_train, y_train)

print("The training set accuracy, with learning rate 0.01, is " + str(gb3.score(x_train,y_train)) )
print("The training set accuracy, with learning rate 0.01, is " + str(gb3.score(x_test,y_test)) )