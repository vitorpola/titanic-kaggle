import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier


train  = pd.read_csv('titanic-data/train.csv')
test  = pd.read_csv('titanic-data/test.csv')

alphabet = ['', 'a','b','c','d','e', 'f','d','e','f','g','h','i','j','k','m', 'n','o','p','q','r','s','t','u','v','w','x','y','z']

train.drop(['Name','Ticket','Cabin'], axis=1, inplace=True)
test.drop(['Name','Ticket','Cabin'], axis=1, inplace=True)

train['Sex'] = train['Sex'].replace(['male','female'], [0,1])
test['Sex'] = test['Sex'].replace(['male','female'], [0,1])

train['Age'].fillna(train['Age'].mean(), inplace=True)
train['Fare'].fillna(train['Fare'].mean(), inplace=True)

test['Fare'].fillna(test['Fare'].mean(), inplace=True)
test['Age'].fillna(test['Age'].mean(), inplace=True)

new_train = pd.get_dummies(train)
new_test = pd.get_dummies(test)

print new_train.head()


x = new_train.drop('Survived', axis=1)
y = new_train['Survived']

mlp = GradientBoostingClassifier(max_depth=10, random_state=1, learning_rate=0.6, n_estimators=100)
print mlp.fit(x,y)
print mlp.score(x,y)

tree = DecisionTreeClassifier(max_depth=4, random_state=1)
tree.fit(x,y)

submission = pd.DataFrame()
submission['PassengerId'] = new_test['PassengerId']
submission['Survived'] = mlp.predict(new_test)
submission.to_csv('submission.csv', index=False)