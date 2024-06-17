"""Transform raw data to train / val datasets """
import argparse
import logging
import pandas as pd
import os
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename='log/preprocess_data.log',
    encoding='utf-8',
    level=logging.DEBUG,
    format='%(asctime)s %(message)s')


IN_FILES = ['data/raw/1_2024-05-22_00-45.csv',
            'data/raw/2_2024-05-22_00-50.csv',
            'data/raw/3_2024-05-22_00-58.csv']

OUT_TRAIN = 'data/proc/train.csv'
OUT_TEST = 'data/proc/test.csv'

TRAIN_SIZE = 0.9
PRICE_THRESHOLD = 30_000_000


def main(args):
    main_dataframe = pd.read_csv(args.input[0], delimiter=';')
    for i in range(1, len(args.input)):
        data = pd.read_csv(args.input[i], delimiter=';')
        df = pd.DataFrame(data)
        main_dataframe = pd.concat([main_dataframe, df], axis=0)

    # Фильтрация строк, где указано метро
    main_dataframe = main_dataframe[pd.notna(main_dataframe['underground'])]

    main_dataframe['url_id'] = main_dataframe['url'].map(lambda x: x.split('/')[-2])
    main_dataframe['first_floor'] = (main_dataframe['floor'] == 1).astype(int)
    main_dataframe['last_floor'] = (main_dataframe['floor'] == main_dataframe['floors_count']).astype(int)

    new_dataframe = main_dataframe[['url_id',
                                    'total_meters',
                                    'first_floor',
                                    'last_floor',
                                    'floors_count',
                                    'price',
                                    'underground']].set_index('url_id')

    new_df = new_dataframe[new_dataframe['price'] < PRICE_THRESHOLD]

    border = int(args.split * len(new_df))
    train_df, val_df = new_df.iloc[:border], new_df.iloc[border:]
    if args.split == 1:
        train_df.to_csv(OUT_TRAIN)
    elif args.split == 0:
        val_df.to_csv(OUT_TEST)
    elif 0 < args.split < 1:
        train_df.to_csv(OUT_TRAIN)
        val_df.to_csv(OUT_TEST)
    else:
        raise ValueError("Wrong split test size!")

    logger.info(f'Write {args.input} to train.csv and test.csv. Train set size: {args.split}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--split', type=float,
                        help='Split data, test relative size, from 0 to 1',
                        default=TRAIN_SIZE)
    parser.add_argument('-i', '--input', nargs='+',
                        help='List of input files',
                        default=IN_FILES)
    args = parser.parse_args()
    main(args)
