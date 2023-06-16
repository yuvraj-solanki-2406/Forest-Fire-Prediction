import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

data = pd.read_csv("C:/Users/HP/Documents/4 Semester/datasets/Forest_fire.csv")
data.head()

X = data.loc[:, data.columns != 'Fire Occurrence']
Y = data['Fire Occurrence']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3,
                                                    random_state=42,
                                                    stratify=Y)

model = LogisticRegression()
model.fit(x_train, y_train)
model_predict = model.predict(x_test)
acc_score = accuracy_score(y_test, model_predict)
print(acc_score)
pickle.dump(model, open('model.pkl', 'wb'))
final_model = pickle.load(open('model.pkl', 'rb'))
