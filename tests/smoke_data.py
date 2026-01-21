
from src.constants import RAW_DATA_PATH
from src.data import load_raw_data

def main():

    df = load_raw_data(RAW_DATA_PATH)
    print(df.shape)
    print(df.head(3))