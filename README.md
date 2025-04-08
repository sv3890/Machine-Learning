Step 1: Importing Required Libraries
You began by importing the necessary libraries:

numpy, pandas, os

librosa and librosa.display for audio processing

matplotlib.pyplot for plotting

Keras and TensorFlow for deep learning model development

Step 2: Loading Metadata
You loaded the UrbanSound8K.csv metadata file that contains file names, class labels, and fold information for audio samples.

This helped identify and locate audio files for feature extraction.

Step 3: Extracting Features (MFCCs)
You used librosa to load and process audio files.

Extracted MFCC (Mel Frequency Cepstral Coefficients) features with shape (40,) for each audio file — which are compact and effective representations of sound.

Stored MFCCs and their corresponding class labels for a subset (e.g., 1000 samples).

Step 4: Data Preparation
Created two lists: X (MFCC features) and Y (labels).

Encoded class labels into one-hot vectors.

Used train_test_split() to divide the dataset into training and testing sets.

Step 5: Building the ANN Model
You created a Sequential ANN model with the following architecture:

Input Layer

Dense layer with 100 neurons

Input shape: (40,)

Activation: ReLU

Dropout: 0.5

Hidden Layer 1

Dense layer with 200 neurons

Activation: ReLU

Dropout: 0.5

Hidden Layer 2

Dense layer with 100 neurons

Activation: ReLU

Dropout: 0.5

Output Layer

Dense layer with number of neurons equal to the number of classes (num_labels)

Activation: Softmax

Step 6: Compiling the Model
Compiled the model using:

Loss Function: categorical_crossentropy

Optimizer: Adam

Evaluation Metric: accuracy

Step 7: Training the Model
Trained the ANN for 20 epochs with a batch size of 32.

Used the validation split or test set to monitor performance.

Step 8: Evaluating the Model
Evaluated model performance on the test set.

Achieved over 81% accuracy.

Step 9: Predictions
Used model.predict() to generate class probabilities for test samples.

Converted these probabilities to class labels using np.argmax.
