"""  Parse data from cian.ru
https://github.com/lenarsaitov/cianparser
"""
import datetime

import cianparser
import pandas as pd
from pathlib import Path
import os

moscow_parser = cianparser.CianParser(location="Москва")

def ensure_directories_exist():
    """
    Ensure that the necessary directories exist.
    """
    directories = ['data', 'data/raw']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        else:
            print(f"Directory already exists: {directory}")


def main():
    """
    Function docstring
    """
    ensure_directories_exist()

    t = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    n_rooms = 1
    csv_path = f'data/raw/{n_rooms}_{t}.csv'
    data = moscow_parser.get_flats(
        deal_type="sale",
        rooms=(n_rooms,),
        with_saving_csv=False,
        additional_settings={
            "start_page": 1,
            "end_page": 2,
            "object_type": "secondary"
        })
    df = pd.DataFrame(data)

    df.to_csv(csv_path,
              encoding='utf-8',
              index=False)


if __name__ == '__main__':
    main()