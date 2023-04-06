from models.ModelsEnum import Models
from models.ModelSelector import select_model
import torch
def test_unet():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_type = Models.UNET_2D
    model = select_model(model_type, num_levels=4, cnn_per_level=2,  input_channels=1, output_channels=1, start_filters=32, kernel_size=3).to('cuda')
    model2 = select_model(model_type, num_levels=4, cnn_per_level=2,  input_channels=1, output_channels=8, start_filters=32, kernel_size=3).to('cuda')
    # Batch, channels, wdith and heiht
    test_input = torch.rand(1, 1, 168, 168, dtype=torch.float32).to(device)
    assert model(test_input).shape == (1, 1, 168, 168)
    assert model2(test_input).shape == (1, 8, 168, 168)