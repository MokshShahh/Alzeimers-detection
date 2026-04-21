# Alzheimer's Detection

This project implements a deep learning system for the detection of Alzheimer's disease using MRI scans from the OASIS-2 longitudinal dataset. It features a robust preprocessing pipeline for medical imaging and a classification model based on the EfficientNet architecture.

## Project Structure

- dataset: Training and validation data splits.
- demented: Preprocessed MRI volumes for demented subjects.
- non-demented: Preprocessed MRI volumes for non-demented subjects.
- logs: CSV files containing training and validation metrics.
- saved_models: Directory for stored model checkpoints.
- Skull-processing-no-pca.ipynb: Pipeline for demographic data cleaning and MRI preprocessing.
- efficientnet.ipynb: Model definition, training loop, and evaluation scripts.

## Preprocessing Pipeline

The preprocessing workflow transforms raw NIfTI scans into standardized formats suitable for neural network training. The pipeline includes:

1. Brain Extraction: Utilizing ANTs for precise skull stripping and brain masking.
2. Contrast Enhancement: Applying CLAHE (Contrast Limited Adaptive Histogram Equalization) to improve feature visibility.
3. Noise Reduction: Sequential application of Gaussian and Median filters.
4. Denoising: Implementation of Non-Local Means Denoising for artifact removal.
5. Volume Preparation: Extraction of relevant 3D slices and conversion to NumPy format.

## Model Architecture

The classification system is built upon the EfficientNet-B2 architecture, optimized for medical image analysis.

- Backbone: Pretrained EfficientNet-B2.
- Custom Head: A specialized classifier consisting of linear layers, ReLU activations, and Dropout for regularization.
- Training: Utilizes Adam optimizer with Cross-Entropy Loss and automated mixed-precision (AMP) for efficiency.
- Early Stopping: Monitors validation loss to prevent overfitting during training sessions.

## Requirements

The following libraries are required to run the preprocessing and training scripts:

- Python 3.12 or higher
- PyTorch and torchvision
- nibabel
- numpy
- opencv-python
- antspyx
- SimpleITK
- scikit-image
- tqdm
- matplotlib

## Usage

### Data Preprocessing
Execute the Skull-processing-no-pca.ipynb notebook to:
- Filter and clean demographic CSV data.
- Process raw MRI files through the brain extraction and filtering pipeline.
- Save preprocessed volumes as NPY files in the categorized directories.

### Model Training and Evaluation
Execute the efficientnet.ipynb notebook to:
- Load the preprocessed datasets using custom data loaders.
- Train the modified EfficientNet-B2 model.
- Monitor metrics via the logging system.
- Evaluate the final model on validation and test sets.

## License

This project is developed for research and educational purposes in medical imaging and diagnostics.
