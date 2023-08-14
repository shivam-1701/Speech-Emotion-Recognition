# Speech Emotion Recognition using RAVDESS Dataset

![Screenshot 2023-08-14 124335](https://github.com/shivam-1701/Speech-Emotion-Recognition/assets/129766853/22c7f680-6f5b-4eb5-a3be-97c7031533c3)

## Table of Contents

- [About the Project](#about-the-project)
  - [Dataset](#dataset)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Usage](#usage)
- [Model](#model)
- [Results](#results)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)

## About the Project

This project focuses on Speech Emotion Recognition (SER) using the RAVDESS dataset. The goal is to develop a machine learning model that can accurately predict emotions from speech audio clips. The Flask web application allows users to upload audio files and receive predicted emotion labels in real-time.

### Dataset

The [RAVDESS dataset](https://zenodo.org/record/1188976#.YK6-IK_0lQI) (Ryerson Audio-Visual Database of Emotional Speech and Song) is a comprehensive collection of speech and song recordings. It consists of 7356 files covering 24 actors, each performing different emotions.

## Getting Started

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/your-project.git
2. cd your-project
3. pip install -r requirements.txt
4. python app.py
   Open your web browser and go to http://localhost:5000 to access the application.
   Upload an audio file and click "Submit" to receive the predicted emotion label.

## Model
The model is based on a deep learning architecture using TensorFlow and Keras. It processes audio clips to extract Mel-Frequency Cepstral Coefficients (MFCCs) as features, which are then fed into the trained model for emotion prediction.

## Results
The model achieves [mention the accuracy or evaluation metric results] on the validation set. Sample output and visualizations can be found in the results directory.

![Screenshot 2023-08-14 124444](https://github.com/shivam-1701/Speech-Emotion-Recognition/assets/129766853/6dd3e059-1c1b-4616-a638-78807a60126f)

## Future Work
Improve model performance with more advanced architectures and data augmentation.
Expand the web application with additional features like real-time recording and emotion visualization.

## Contributing
Contributions are welcome! If you find any issues or have suggestions for improvements, please feel free to open an issue or create a pull request.

## Acknowledgments
Mention any references, tutorials, or libraries that were helpful in your project.
Give credit to the authors of the RAVDESS dataset.

