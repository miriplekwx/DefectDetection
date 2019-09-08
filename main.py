from model import Unet
import modelUsing
from modelUsing import *



if __name__ == "__main__":
    model = Unet("resnet18", encoder_weights="imagenet", classes=4, activation=None)
    print(model)
    model_trainer = Trainer(model)
    model_trainer.start()