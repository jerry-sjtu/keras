from keras.layers import Dense, Activation
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Merge
from sklearn.preprocessing import MinMaxScaler

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
    "occupation", "relationship", "race", "gender", "capital_gain", "capital_loss",
    "hours_per_week", "native_country", "income_bracket"
]

LABEL_COLUMN = "label"

CATEGORICAL_COLUMNS = [
    "workclass", "education", "marital_status", "occupation", "relationship",
    "race", "gender", "native_country"
]

CONTINUOUS_COLUMNS = [
    "age", "education_num", "capital_gain", "capital_loss", "hours_per_week"
]


def load(filename):
    with open(filename, 'r') as f:
        skiprows = 1 if 'test' in filename else 0
        df = pd.read_csv(
            f, names=COLUMNS, skipinitialspace=True, skiprows=skiprows, engine='python'
        )
        df = df.dropna(how='any', axis=0)
    return df


def preprocess(df):
    df[LABEL_COLUMN] = df['income_bracket'].apply(lambda x: ">50K" in x).astype(int)
    df.pop("income_bracket")
    y = df[LABEL_COLUMN].values
    df.pop(LABEL_COLUMN)

    df = pd.get_dummies(df, columns=[x for x in CATEGORICAL_COLUMNS])

    # TODO: select features for wide & deep parts

    # TODO: transformations (cross-products)
    # from sklearn.preprocessing import PolynomialFeatures
    # X = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False).fit_transform(X)

    df = pd.DataFrame(MinMaxScaler().fit_transform(df), columns=df.columns)

    X = df.values
    return X, y


def main():
    df_train = load('/Users/qiangwang/.keras/datasets/census_data/adult.data')
    df_test = load('/Users/qiangwang/.keras/datasets/census_data/adult.test')
    df = pd.concat([df_train, df_test])
    train_len = len(df_train)

    X, y = preprocess(df)
    X_train = X[:train_len]
    y_train = y[:train_len]
    X_test = X[train_len:]
    y_test = y[train_len:]

    model = Sequential()
    model.add(Dense(input_dim=X_train.shape[1], activation='sigmoid'))
    model.add(Dense(input_dim=100, units=10, activation='sigmoid'))
    model.add(Dense(units=64, input_dim=100))
    model.add(Activation('relu'))
    model.add(Dense(units=10))
    model.add(Activation('softmax'))
    model.compile(loss='binary_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=1, batch_size=32)

    loss, accuracy = model.evaluate([X_test, X_test], y_test)
    print('\n', 'test accuracy:', accuracy)

    model.save('/Users/qiangwang/.keras/datasets/census_model/base.h5')
    with open('/Users/qiangwang/.keras/datasets/census_model/model.json', 'w') as out:
        out.write(model.to_json() + '\n')
    model.save_weights('/Users/qiangwang/.keras/datasets/census_model/weight.h5')


if __name__ == '__main__':
    main()
