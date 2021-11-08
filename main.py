'''
Author: your name
Date: 2021-11-06 09:36:32
LastEditTime: 2021-11-08 22:41:44
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: \PetFinderNew\main.py
'''
import time
from comet_ml import Experiment
import torch.optim as optim
from torchsummary import summary
from Project import Project
from data import get_dataloaders
from data.transformation import train_transform, val_transform
from models import MyCNN, resnet18
from utils import device, show_dl
from poutyne.framework import Model
from poutyne.framework.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from callbacks import CometCallback
from logger import logging
import yaml
import argparse
from .data.transformation.default_tranform import default_transforms
from data.MyDataset import *
from torch.utils.data import DataLoader, random_split

def getConfig():
    description = "you should add those parameter"                    
    parser = argparse.ArgumentParser(description=description)       
    parser.add_argument('--address',help = "config address", default="configs\\config.yaml")
    args = parser.parse_args()
    with open(args.address) as conf:
        configFile = yaml.safe_load(conf)
    return configFile
    
if __name__ == '__main__':
    # load config file
    conf = getConfig()
    # project = Project()
    # our hyperparameters
    params = {
        'lr': 0.001,
        'batch_size': 64,
        'epochs': 10,
        'model': 'resnet18-finetune'
    }
    logging.info(f'Using device={device} 🚀')
    # everything starts with the data
    # train_dl, val_dl, test_dl = get_dataloaders(
    #     project.data_dir / "train",
    #     project.data_dir / "val",
    #     val_transform=val_transform,
    #     train_transform=train_transform,
    #     batch_size=params['batch_size'],
    #     pin_memory=True,
    #     num_workers=4,
    # )
    # is always good practice to visual
    ######DataLoader must be in main method#######
    train_dataset, val_dataset = createDataset(conf['root_path'], default_transforms, conf['image_size'])
    ##############################################
    show_dl(test_dl)
    # define our comet experiment
    experiment = Experiment(api_key="YOU_KEY",
                            project_name="dl-pytorch-template", workspace="francescosaveriozuppichini")
    experiment.log_parameters(params)
    # create our special resnet18
    cnn = resnet18(2).to(device)
    # print the model summary to show useful information
    logging.info(summary(cnn, (3, 224, 244)))
    # define custom optimizer and instantiace the trainer `Model`
    optimizer = optim.Adam(cnn.parameters(), lr=params['lr'])
    model = Model(cnn, optimizer, "cross_entropy",
                  batch_metrics=["accuracy"]).to(device)
    # usually you want to reduce the lr on plateau and store the best model
    callbacks = [
        ReduceLROnPlateau(monitor="val_acc", patience=5, verbose=True),
        ModelCheckpoint(str(project.checkpoint_dir /
                            f"{time.time()}-model.pt"), save_best_only="True", verbose=True),
        EarlyStopping(monitor="val_acc", patience=10, mode='max'),
        CometCallback(experiment)
    ]
    model.fit_generator(
        train_dl,
        val_dl,
        epochs=params['epochs'],
        callbacks=callbacks,
    )
    # get the results on the test set
    loss, test_acc = model.evaluate_generator(test_dl)
    logging.info(f'test_acc=({test_acc})')
    experiment.log_metric('test_acc', test_acc)