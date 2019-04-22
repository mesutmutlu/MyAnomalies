import pandas as pd
import sys

if __name__ == "__main__":
    sys.stdout.buffer.write(chr(9986).encode('utf8'))
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    movies_df = pd.read_csv(r'C:\datasets\the-movies-dataset\movies_metadata.csv', low_memory=False)
    print(movies_df.head())
    cred_df = pd.read_csv(r'C:\datasets\the-movies-dataset\credits.csv')
    print(cred_df.head())
    key_df = pd.read_csv(r'C:\datasets\the-movies-dataset\keywords.csv')
    print(key_df.head())
    lang_codes_df = pd.read_csv(r'C:\datasets\the-movies-dataset\language-codes-full.csv')
    print(lang_codes_df.head())