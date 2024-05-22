# Deep Learning-based Crop Classification using CNN Architecture

## Overview

This project implements a Convolutional Neural Network (CNN) for crop classification using TensorFlow and Keras. The model is trained on a dataset of crop images to classify different types of crops accurately. This repository includes the complete code for data preprocessing, model training, evaluation, and prediction.

## Author

**Rugved Jalit**

## Table of Contents

- [Installation](#installation)

- [Dataset](#dataset)

- [Usage](#usage)

- [Model Architecture](#model-architecture)

- [Training](#training)

- [Evaluation](#evaluation)

- [Prediction](#prediction)

- [Results](#results)

- [Contributing](#contributing)

- [License](#license)

## Installation

To run this project, you need to have Python installed along with the required libraries. You can install the necessary packages using pip:

```bash

pip install tensorflow pandas numpy seaborn matplotlib

```

Additionally, if you are using Google Colab, you need to mount your Google Drive to access the dataset:

```python

from google.colab import drive

drive.mount('/content/drive')

```

## Dataset

The dataset used in this project is stored in a directory structure where each subdirectory represents a class of crops. The images are loaded and preprocessed using TensorFlow's `image_dataset_from_directory` method.

## Usage

1\. **Load the dataset:**

    ```python

    dataset = tf.keras.preprocessing.image_dataset_from_directory(

        '/content/drive/MyDrive/cotton/Working',

        seed=19,

        shuffle=True,

        image_size=(256, 256),

        batch_size=28

    )

    ```

2\. **Split the dataset:**

    ```python

    train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)

    ```

3\. **Define and compile the model:**

    ```python

    model = models.Sequential([

        # Model layers

    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    ```

4\. **Train the model:**

    ```python

    history = model.fit(train_ds, validation_data=val_ds, epochs=10)

    ```

5\. **Evaluate the model:**

    ```python

    scores = model.evaluate(test_ds)

    ```

## Model Architecture

The CNN model consists of several convolutional layers followed by max-pooling layers and dense layers. Here's an overview of the architecture:

- Input Layer

- Convolutional Layers with ReLU activation

- Max Pooling Layers

- Flatten Layer

- Dense Layers with ReLU activation

- Output Layer with Softmax activation

## Training

The model is trained with the following parameters:

- **Batch Size:** 28

- **Image Size:** 256x256

- **Channels:** 3 (RGB)

- **Epochs:** 10    # Due to system compatibility issues, set EPOCHS to 10. Adjust as needed.                 

The training and validation accuracy and loss are plotted to monitor the model's performance.

## Evaluation

The model is evaluated on the test set to determine its accuracy. The evaluation metrics include accuracy and loss.

## Prediction

A function `predict` is defined to make predictions on new images. It returns the predicted class and confidence level.

```python

def predict(model, img):

    # Function to predict class and confidence

```

## Results

The results of the training process, including accuracy and loss plots, are displayed. Sample predictions on test images are also shown with actual and predicted labels along with confidence levels.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any changes or improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

Feel free to explore the code and modify it according to your needs. If you have any questions, please contact the author.

Happy Coding! 🌾
