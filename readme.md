# PET Bottle Detection

This repository contains code for detecting if a bottle is made of PET (Polyethylene Terephthalate) using a Convolutional Neural Network (CNN) model built with TensorFlow and Keras. It includes scripts for training the model and for using the model to make predictions on new images.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Training the Model](#training-the-model)
- [Using the Model](#using-the-model)
- [File Structure](#file-structure)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Model Training**: Build and train a CNN to classify PET vs. non-PET bottles.
- **Model Inference**: Use the trained model to predict whether a given bottle is made of PET.
- **Image Preprocessing**: Includes data augmentation and preprocessing for robust training.

## Prerequisites

Ensure you have the following installed:

- Python 3.x
- TensorFlow
- OpenCV
- NumPy
- Matplotlib

You can install the required packages using pip:

```bash
pip install tensorflow opencv-python numpy matplotlib
```

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/pet-bottle-detection.git
    cd pet-bottle-detection
    ```

2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Training the Model

1. **Prepare Dataset**: Ensure your dataset is organized as follows:

    ```
    dataset/
    ├── train/
    │   ├── PET/
    │   └── Non-PET/
    └── validation/
        ├── PET/
        └── Non-PET/
    ```

2. **Run the Training Script**:

    ```bash
    python train_model.py
    ```

    This script will train the CNN model and save it as `pet_bottle_classifier.h5`.

## Using the Model

1. **Ensure Camera is Connected**: Verify that your camera is working and accessible.

2. **Run the Inference Script**:

    ```bash
    python check_bottle.py
    ```

    This script will use the trained model to classify bottles captured from the webcam. It will print `1` if the bottle is PET and `0` otherwise.

## File Structure

- `train_model.py`: Script for training the CNN model.
- `check_bottle.py`: Script for using the trained model to classify bottles.
- `dataset/`: Directory containing the dataset for training and validation.
- `requirements.txt`: List of Python dependencies.
- `pet_bottle_classifier.h5`: The trained model file (generated after training).

## Contributing

Contributions are welcome! Please submit issues and pull requests to help improve the project.

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.