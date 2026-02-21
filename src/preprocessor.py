from src.db import read_sql_train, read_sql_test
import logging
import numpy as np

logger = logging.getLogger(__name__)

QUERY_TRAIN = "SELECT * FROM application_train"
QUERY_TEST = "SELECT * FROM application_test"

def load_data_train():
    return read_sql_train(QUERY_TRAIN)

def load_data_test():
    return read_sql_test(QUERY_TEST)

def preprocessing(df):
    df = df.copy()
    
    # DAYS_BIRTH -> YEARS
    df['DAYS_BIRTH_IN_YEAR'] = -df['DAYS_BIRTH'] / 365.25
    df = df.drop(columns=['DAYS_BIRTH'])
    
    # DAYS_EMPLOYED
    df['IS_UNEMPLOYED'] = (df['DAYS_EMPLOYED'] == 365243).astype(int)
    df.loc[df['DAYS_EMPLOYED'] == 365243, 'DAYS_EMPLOYED'] = np.nan

    # EMPLOYED_YEARS
    df['EMPLOYED_YEARS'] = (-df['DAYS_EMPLOYED'] / 365.25).clip(lower=0)
    df = df.drop(columns=['DAYS_EMPLOYED'])

    # DAYS_ID_PUBLISH -> YEARS
    df['DAYS_ID_PUBLISH_IN_YEAR'] = -df['DAYS_ID_PUBLISH'] / 365.25
    df = df.drop(columns=['DAYS_ID_PUBLISH'])

    # CODE_GENDER
    df['CODE_GENDER'] = df['CODE_GENDER'].replace('XNA', np.nan)
    
    # NAME_FAMILY_STATUS
    df['NAME_FAMILY_STATUS'] = df['NAME_FAMILY_STATUS'].replace('Unknown', np.nan)

    return df

def build_train_test():
    df_application_train = load_data_train()
    df_application_test  = load_data_test() 

    X_train = preprocessing(df_application_train.drop(columns=['TARGET']))
    y_train = df_application_train['TARGET']
    X_test = preprocessing(df_application_test)

    # проверка размерности признаков после сплит
    if X_train.shape[1] != X_test.shape[1]:
        logger.error('Количество признаков после сплит не совпадает, проверьте датасеты')
    else:
        logger.info('Проверка размерности признаков после сплит пройдена')

    # проверка имен/состава колонок
    if set(X_train.columns) != set(X_test.columns):
        logger.error('Имена признаков после сплит не совпадают, проверьте датасеты')
    else:
        logger.info('Проверка имен колонок после сплит пройдена')
    
    return X_train, y_train, X_test

# Catboost preprocessing: categorial value
def prepare_for_catboost(X):
    X = X.copy()
    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    for col in cat_cols:
        X[col] = X[col].fillna("NA")
    
    # проверка (в cat_cols не должны остатся nan)
    if X[cat_cols].isna().sum().sum() == 0:
        logger.info('Кат. признаки в X_train подготовлены для обучения в CatBoost')
    else:
        logger.error('Проверьте кат. признаки (X_train), найдены nan значения')

    if X[cat_cols].isna().sum().sum() == 0:
        logger.info('Кат. признаки в X_test подготовлены')
    else:
        logger.error('Проверьте кат. признаки (X_test), найдены nan значения')

    return X, cat_cols