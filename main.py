import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle as pickle


def get_clean_data():
    data = pd.read_csv("C:/Users/erfan/PycharmProjects/pythonProject1/SteamLit-Cancer-Predictor/data/breast cancer.csv")
    data.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    return data


def create_model(data):
    X = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']

    # scaling the data for the model to perform correctly
    data_scaler = StandardScaler()
    X = data_scaler.fit_transform(X)

    # split data into train & test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # test the model
    y_prediction = model.predict(X_test)

    # accuracy of the model
    accuracy = accuracy_score(y_test, y_prediction)
    print('Accuracy of our model: ', accuracy)
    report = classification_report(y_test, y_prediction)
    print('Full report of our model: \n', report)

    return model, data_scaler


def main():
    data = get_clean_data()
    model, data_scaler = create_model(data)
    with open("C:/Users/erfan/PycharmProjects/pythonProject1/SteamLit-Cancer-Predictor/model/model.pkl", 'wb') as f:
        pickle.dump(model, f)
    with open("C:/Users/erfan/PycharmProjects/pythonProject1/SteamLit-Cancer-Predictor/model/data_scaler.pkl", 'wb') as f:
        pickle.dump(data_scaler, f)


if __name__ == '__main__':
    main()






