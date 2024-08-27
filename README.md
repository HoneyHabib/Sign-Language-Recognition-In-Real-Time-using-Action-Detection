# Sign-Language-recognition-in-Real-Time-using-action-detection
# Sign Language Recognition Using Action Detection with LSTM

Welcome to the **Sign Language Recognition Using Action Detection** repository! This project utilizes Long Short-Term Memory (LSTM) neural networks built with TensorFlow/Keras to recognize sign language gestures through action detection.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Getting Started](#getting-started)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Model Building](#model-building)
  - [Evaluation and Prediction](#evaluation-and-prediction)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Project Overview

This repository contains a machine learning model designed to recognize sign language gestures using action detection. The model employs LSTM networks to analyze sequences of video frames and classify gestures based on the temporal dynamics of the actions.

## Features

- **LSTM Model**: Utilizes LSTM networks to capture temporal patterns in gesture sequences.
- **Action Detection**: Processes video frames to detect and interpret sign language gestures.
- **Performance Metrics**: Evaluates model accuracy and gesture classification performance.
- **Visualization Tools**: Visualizes model predictions and recognition results.

## Technologies Used

- **Python**: Programming language for implementing the model.
- **TensorFlow/Keras**: Libraries for building and training the LSTM neural network.
- **OpenCV**: For video processing and frame extraction.

## Getting Started

To get started with this project, follow these steps:

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/sign-language-recognition-action-detection.git
   ```

2. **Navigate to the Project Directory**

   ```bash
   cd sign-language-recognition-action-detection
   ```

3. **Install Dependencies**

   You will need to install the following Python packages:

   - TensorFlow
   - Mediapipe
   - OpenCV
   - sklearn
   - matplotlib

   Install these packages using `pip`:

   ```bash
   pip install tensorflow keras opencv-python numpy
   ```

4.  **Run the Notebook**

   Open `Sign_Language_Recognition_LSTM.ipynb` using Jupyter Notebook or JupyterLab and follow the instructions to train and test the model.

## Usage

### Data Preparation

First, extract frames from videos and prepare them for model training:


### Model Building

Build and compile the LSTM model for gesture recognition:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir= log_dir)

# Define model
model = Sequential()
model.add(LSTM(64, return_sequences= True, activation = 'relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences= True, activation = 'relu'))
model.add(LSTM(64, return_sequences = False, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(actions.shape[0], activation ='softmax'))

# Compile model
model.compile(optimizer= 'Adam', loss= 'categorical_crossentropy', metrics=['categorical_accuracy'])
```

### Evaluation and Prediction

Evaluate the model and make predictions:

```python
# Train model

model.fit(x_train, y_train, epochs=2000, callbacks=[tb_callback])

# Prediction
res = model.predict(x_test)
actions[np.argmax(res[0])]
actions[np.argmax(y_test[0])]

# Display results

plt.figure(figsize=(18,18))
plt.imshow(prob_viz(res, actions, image, colors))
```
![sign language recognition](https://github.com/user-attachments/assets/8ff302fb-639f-4606-989b-7f41a0c70e43)


## Contributing

Contributions are welcome! If you have suggestions, improvements, or fixes, please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or feedback, please reach out to [your email address] or open an issue in the repository.

Happy coding! 🚀

---

Feel free to adjust or expand the snippets based on your project's specifics and data handling processes.
