import torch
from libs.data import Data
from libs.opt import CFG

args = CFG()

dataset = Data(args)

model_name = args.model_name.lower()
if model_name == "baseline":
    class_name = "BaselineMethod"
elif model_name == "c2ae":
    class_name = "C2AEMethod"
elif model_name == "csgrl":
    class_name = "csgrl"
elif model_name == "opengan":
    class_name = "OpenGAN"
else:
    raise ValueError(f"Unknown model_name '{args.model_name}'")

exec(
    "from methods.{model_name}.model import {class_name}".format(
        model_name=model_name,
        class_name=class_name,
    )
)
exec("model = {class_name}(args)".format(class_name=class_name))

train_loader, test_loader = dataset.get_dataloader()
exec("model.{}(train_loader, test_loader)".format(args.phase))
