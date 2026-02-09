import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_selector
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def filter_data(df):
    df_filter = df.copy()
    columns_to_drop = [
        'id',
        'url',
        'region',
        'region_url',
        'price',
        'manufacturer',
        'image_url',
        'description',
        'posting_date',
        'lat',
        'long'
    ]
    return df_filter.drop(columns_to_drop, axis = 1)

def remove_outliers(df):
    df_remove = df.copy()
    q25 = df_remove['year'].quantile(0.25)
    q75 = df_remove['year'].quantile(0.75)
    iqr = q75 - q25
    boundaries = (q25 - 1.5 * iqr, q75 + 1.5 * iqr)

    df_remove.loc[df_remove['year'] < boundaries[0], 'year'] = round(boundaries[0])
    df_remove.loc[df_remove['year'] > boundaries[1], 'year'] = round(boundaries[1])

    return df_remove


def new_features(df):
    df_features = df.copy()

    def short_model(df_features):
        if not pd.isna(df_features):
            return df_features.lower().split(' ')[0]
        else:
            return df_features

    df_features.loc[:, 'short_model'] = df_features['model'].apply(short_model)
    df_features.loc[:, 'age_category'] = df_features['year'].apply(lambda x: 'new' if x > 2013 else ('old' if x < 2006 else 'average'))

    return df_features


def main():
    print('Price category Prediction Pipeline')

    df = pd.read_csv('data/homework.csv')

    X = df.drop(['price_category'], axis=1)
    y = df['price_category']

    numerical_features = make_column_selector(dtype_include=['int64', 'float64'])
    categorical_features = make_column_selector(dtype_include=object)

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = Pipeline(steps=[
        ('filter', FunctionTransformer(filter_data)),
        ('outliers', FunctionTransformer(remove_outliers)),
        ('new_features', FunctionTransformer(new_features))
    ])

    preprocessor_column = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, numerical_features),
        ('categorical', categorical_transformer, categorical_features)
    ])

    models = (
        LogisticRegression(solver='liblinear'),
        RandomForestClassifier(),
        SVC()
    )

    best_score = .0
    best_pipe = None
    for model in models:
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('transform_column', preprocessor_column),
            ('classifier', model)
        ])
        score = cross_val_score(pipe, X, y, cv=4, scoring='accuracy')
        print(f'model: {type(model).__name__}, acc_mean: {score.mean():.4f}, acc_std: {score.std():.4f}')

        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe

    print(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, accuracy: {best_score:.4f}')
    joblib.dump(best_pipe, 'price_category_pipe.pkl')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()


