#Sign Language Detector

This project is designed to translate sign language into text or speech using machine learning and Python. It leverages MediaPipe, hand landmarks, and a Random Forest classifier to recognize and interpret sign language gestures in real-time.

#Overview

Sign language is a vital communication method for individuals with hearing impairments. This project aims to bridge communication gaps by providing a tool that detects and translates sign language gestures into readable text or spoken words in real time.

#Features

Real-time Gesture Recognition: Utilizes MediaPipe to track hand movements and detect gestures.

Landmark-Based Analysis: Extracts key hand points to differentiate between various sign language gestures.

Machine Learning Model: Implements a Random Forest classifier to accurately classify and translate gestures into corresponding text or speech output.

How to Use

Installation

Clone the repository:

git clone https://github.com/arpit8929/sign-language-detector.git

Navigate to the project directory:

cd sign-language-detector

Install the required dependencies:

pip install -r requirements.txt

Running the Application

To start the sign language detector, run the following command:

python app.py

Interacting with the Translator

Enable the camera to begin real-time sign recognition.

Perform sign language gestures in front of the camera for interpretation.

Project Report

For an in-depth explanation of the methodology, findings, and performance analysis, refer to the Project Report available in the repository.

Contributions

Contributions are welcome! If you’d like to improve this project, feel free to open issues, submit pull requests, or share your ideas for enhancements.