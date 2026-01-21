
from __future__ import annotations


from src.loaders import make_loaders
from src.model import SkillPriorityNet


def main():

    loader = make_loaders()

    model = SkillPriorityNet(input_dim=loader.meta['num_features'], num_classes=loader.meta['num_classes'])

    xb, yb = next(iter(loader.train_loader))
    logits = model(xb)
    print(logits.shape)


if __name__ == "__main__":
    main()





