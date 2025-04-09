# Lagrangian Coherent Eddy Detection and Prediction

This repository contains the code implementation for detecting and predicting Lagrangian coherent eddies in the Gulf of Mexico using machine learning techniques. The project combines multiple satellite observations (ADT, SST, and Chlorophyll-a) to identify and predict the lifetime of Loop Current Eddies (LCEs) and Loop Current Frontal Eddies (LCFEs).

## Project Structure

### Core ML Components (`proj_ai/`)
- `Training.py`: Implements the main training loop for PyTorch models, including checkpointing, tensorboard logging, and early stopping
- `Generators.py`: Contains the `EddyDataset` class for loading and preprocessing eddy tracking data from multiple satellite sources

### Model Architectures (`models/` and `ai_common/models/`)
- `Models2D.py`: Defines the core neural network architectures including:
  - `MultiStreamUNet`: A modified U-Net that processes multiple input streams (ADT, SST, Chlorophyll-a)
  - `EncoderDecoderBlock`: Building block for encoder-decoder architectures
- `modelBuilder2D.py`: Factory functions for creating various 2D model architectures
- `modelBuilder3D.py`: Factory functions for creating 3D model architectures

### Data Processing and Visualization (`eoas_pyutils/`)
- `io_utils/io_netcdf.py`: Utilities for reading NetCDF satellite data files
- `viz_utils/eoa_viz.py`: Visualization tools for geospatial data using matplotlib and cartopy
  - Supports multiple data formats
  - Customizable map projections and backgrounds
  - Automatic colormap selection

## Data Requirements

The model expects the following satellite data:
- Absolute Dynamic Topography (ADT)
- Sea Surface Temperature (SST)
- Chlorophyll-a concentration

Data should be in NetCDF format with consistent temporal and spatial resolution.

## Citation

If you use this code in your research, please cite:
[Paper citation details]

## License

[License details]

## Contributors

[List of contributors]

## Acknowledgments

This work was supported by [funding/support details]
