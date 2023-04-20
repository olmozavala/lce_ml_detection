from enum import Enum
from models.Models2D import UNet
from models.ModelsEnum import Models
# Enum of model architectures

def select_model(model_option, **kwargs):
    '''
    Selects a model based on the model_option
    :param model_option:  A ModelOptions enum
    '''
    if model_option == Models.UNET_2D:
        return UNet(**kwargs)
    raise ValueError('Invalid model option')
