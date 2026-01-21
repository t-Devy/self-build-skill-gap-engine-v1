
from src.constants import RAW_DATA_PATH
from src.features import build_features
from src.data import load_raw_data

def main():
    df = load_raw_data(RAW_DATA_PATH)
    print(build_features(df))


if __name__ == "__main__":
    main()