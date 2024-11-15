# Speech Command Recognition using ANNs And CNNs

| Name                 | ID          |
| -------------------- | ----------- |
| Ahmed Ashraf Mohamed | 2103134     |

# Project Objective

The objective of this project is to build and evaluate a speech command recognition model using Artificial Neural Networks (ANNs) and Convolutional Neural Networks (CNNs).
This models are designed to recognize specific spoken commands from audio input and classify them into predefined categories.
And Compare the models to determine which is a better model.

# Data-Set

[Google Command Speech](https://www.kaggle.com/datasets/neehakurelli/google-speech-commands/code), This is a set of one-second .wav audio files, each containing a single spoken
English word. These words are from a small set of commands, and are spoken by a variety of different speakers. 

## Preprocessing

- Loading and Normalzing
  - During Loading the audio, we normalize the audio between -1 and 1 which will increase the number of frames
- Feature Extraction
  - Extracting MFCC (n_mfcc = 40) and then normalizing it between -1 and 1

# Running the Code

Download the [Google Command Speech](https://www.kaggle.com/datasets/neehakurelli/google-speech-commands/code) data-set, and use `augment.py` to augment the data-set

You can just run the `main.ipynb` notebook where all the processes are done. There is a `utils` and `model` folders that contains helper functions to make the code in the notebook easier to read and follow along.

## Usage of `augment.py`

```
python augment.py --input /path/to/data-set --noise /path/to/noise-dataset --output /path/to/output-dir --sampling sampling_rate
```

You can skip the step of augmenting the data-set.

# Dependencies

- `python = 3.12.7`
- `numpy = 2.0.2`
- `matplotlib = 3.9.2`
- `librosa = 0.10.2`
- `scikit-learn = 1.5.2`
- `tensorflow = 2.18.0`
