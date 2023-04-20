from enum import Enum
# Enum of model architectures
class Models(Enum):
    UNET_2D = 1
    UNET_2D_MultiStream = 2
    HALF_UNET_2D_Classification = 3
