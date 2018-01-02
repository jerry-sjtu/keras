import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Merge, Embedding, Flatten
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

    edu_col = df['education'].astype('category').cat.codes
    df = pd.get_dummies(df, columns=[x for x in CATEGORICAL_COLUMNS])
    # for x in df.columns:
    #     print(x)
    # print(df[:10]['education_1st-4th'])
    # print(df[:10]['education_5th-6th'])
    # print(df[:10]['education_9th'])

    # print(edu_col)
    # print(type(edu_col))
    # print(edu_col.unique())


    # TODO: select features for wide & deep parts

    # TODO: transformations (cross-products)
    # from sklearn.preprocessing import PolynomialFeatures
    # X = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False).fit_transform(X)

    df = pd.DataFrame(MinMaxScaler().fit_transform(df), columns=df.columns)

    X = df.values # numpy.ndarray
    # print(type(X))
    # print(X.shape)
    edu_col = edu_col.values
    # print(type(edu_col))
    # print(edu_col.shape)
    return X, y, edu_col


def main():
    df_train = load('/Users/qiangwang/.keras/datasets/census_data/adult.data')
    df_test = load('/Users/qiangwang/.keras/datasets/census_data/adult.test')
    df = pd.concat([df_train, df_test])
    train_len = len(df_train)

    X, y, edu = preprocess(df)

    X_train = X[:train_len]
    y_train = y[:train_len]
    edu_train = edu[:train_len]
    X_test = X[train_len:]
    y_test = y[train_len:]
    edu_test = edu[train_len:]

    wide = Sequential()
    wide.add(Dense(1, input_dim=X_train.shape[1]))

    embed = Sequential()
    # input_length=2 可以是序列行为。。。
    embed.add(Embedding(input_dim=16, output_dim=8, input_length=1))
    # embed.add(Embedding(input_dim=16, output_dim=8, input_length=2))
    embed.add(Flatten())

    deep = Sequential()
    deep.add(Dense(input_dim=X_train.shape[1], units=100, activation='relu'))
    # deep.add(Dense(100, activation='relu'))
    # deep.add(Dense(50, activation='relu'))
    # deep.add(Dense(1, activation='sigmoid'))
    model1 = Sequential()
    model1.add(Merge([deep, embed], mode='concat', concat_axis=1))
    model1.add(Dense(100, activation='relu'))
    model1.add(Dense(50, activation='relu'))

    model = Sequential()
    model.add(Merge([wide, model1], mode='concat', concat_axis=1))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer='rmsprop',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.fit([X_train, X_train, edu_train], y_train, epochs=5, batch_size=32)
    # model.fit([X_train, X_train], y_train, epochs=1, batch_size=32)

    loss, accuracy = model.evaluate([X_test, X_test, edu_test], y_test)
    print('\n', 'test accuracy:', accuracy)

    # deep = Sequential()
    # deep.add(Dense(input_dim=X_train.shape[1], output_dim=100, activation='relu'))
    # deep.add(Dense(100, activation='relu'))
    # deep.add(Dense(50, activation='relu'))
    # # 加上这一层相当于是多模型的融合，而不是deed and wide模型，去掉这一层以后效果变好了
    # # deep.add(Dense(1, activation='sigmoid'))
    # model = Sequential()
    # model.add(Merge([wide, deep], mode='concat', concat_axis=1))
    # model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer='rmsprop',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # epochs太小，结果不稳定
    model.fit([X_train, X_train], y_train, epochs=5, batch_size=32)
    loss, accuracy = model.evaluate([X_test, X_test], y_test)
    print('\n', 'test accuracy:', accuracy)


    model.save('/Users/qiangwang/.keras/datasets/census_model/base.h5')
    with open('/Users/qiangwang/.keras/datasets/census_model/model.json', 'w') as out:
        out.write(model.to_json() + '\n')
    model.save_weights('/Users/qiangwang/.keras/datasets/census_model/weight.h5')


if __name__ == '__main__':
    main()