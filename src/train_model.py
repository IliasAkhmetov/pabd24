"""Train model and save checkpoint"""

import argparse
import logging
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error
from joblib import dump

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename='log/train_model.log',
    encoding='utf-8',
    level=logging.DEBUG,
    format='%(asctime)s %(message)s')

TRAIN_DATA = 'data/proc/train.csv'
MODEL_SAVE_PATH = 'models/catboost_regression_v01.joblib'


def main(args):
    df_train = pd.read_csv(TRAIN_DATA)
    X_train = df_train[['total_meters',
                        'first_floor',
                        'last_floor',
                        'floors_count',
                        'underground'  # добавляем категориальный признак
                        ]]
    y_train = df_train['price']

    # Указание категориальных признаков (номера колонок или имена колонок)
    categorical_features = ['underground']

    # Создание и обучение модели CatBoostRegressor
    model = CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=6,
                              eval_metric='MAE', random_seed=42, verbose=100)
    model.fit(X_train, y_train, cat_features=categorical_features, use_best_model=True)

    # Сохранение модели
    dump(model, args.model)
    logger.info(f'Saved to {args.model}')

    # Оценка модели
    y_pred = model.predict(X_train)
    mae = mean_absolute_error(y_train, y_pred)

    logger.info(f'Mean Absolute Error on train set: {mae:.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model',
                        help='Model save path',
                        default=MODEL_SAVE_PATH)
    args = parser.parse_args()
    main(args)
