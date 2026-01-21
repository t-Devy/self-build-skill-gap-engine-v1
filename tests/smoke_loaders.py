

from src.loaders import make_loaders


def main():

    loader = make_loaders(batch_size=8)
    xb, yb = next(iter(loader.train_loader))
    print(f"Train batch shapes, X={xb.shape}, y={yb.shape}")
    print(f"Features: {loader.meta['num_features']}")
    print(f"Classes: {loader.meta['num_classes']}")
    print("")

    print("Example Training Batches:")
    for i, (xb, yb) in enumerate(loader.train_loader):
        if i >= 3:
            break
        print(f" Batch {i}: X={xb.shape}, y={yb.shape}, y_unique={yb.unique().tolist()}")


if __name__ == "__main__":
    main()
