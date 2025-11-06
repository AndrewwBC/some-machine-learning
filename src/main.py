from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from adaline import file

iris = datasets.load_iris()

features = iris.data[:, [2, 3]][0:100]
target = iris.target[0:100]

features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.3, random_state=1, stratify=target
)

scaler = StandardScaler()
scaler.fit(features_train)
features_train_std = scaler.transform(features_train)
features_test_std = scaler.transform(features_test)

adaline = file.Adaline()
adaline.fit(features_train_std, target_train)

predicted = adaline.predict(features_test_std)
acc = accuracy_score(target_test, predicted)
print(acc)