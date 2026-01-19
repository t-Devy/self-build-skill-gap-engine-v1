
from src.data import load_raw_data
from src.constants import RAW_DATA_PATH, FEATURE_COLS, TARGET_COL
from src.schema import validate_schema


def main():

    df = load_raw_data(RAW_DATA_PATH)
    print(df.shape)
    print(df.head(3))
    print("")

    report = validate_schema(df)

    if report.ok:
        print("Schema OK: True")
        print("")
        for col in FEATURE_COLS:
            feature_bounds = {"min": int(df[col].min()), "max": int(df[col].max())}
            print(f"{col}: {feature_bounds}")

        print("")
        counts = df[TARGET_COL].value_counts()
        for label in counts.index:
            count = counts[label]
            print(f"label: {label}, count: {count}")
    else:
        print("Schema validation failed:")
        for error in report.errors:
            print(f" - {error}")


if __name__ == "__main__":
    main()
