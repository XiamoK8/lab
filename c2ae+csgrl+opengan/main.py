from libs.data import Data
from libs.opt import options


def main():
    args = options().get_args()
    dataset = Data(args)
    model_name = args.model_name.lower()

    exec(f"from methods.{model_name}.model import {args.model_name}")
    exec(f"model = {args.model_name}(args)")

    train_loader, test_loader = dataset.get_dataloader()
    exec(f"model.{args.phase}(train_loader, test_loader)")


if __name__ == "__main__":
    main()
