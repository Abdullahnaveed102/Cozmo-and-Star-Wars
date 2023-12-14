# Cozmo-and-Star-Wars
Cozmo's Star Wars Odyssey - A project integrating computer vision for character recognition, ChatGPT for storytelling, and user queries, enhancing Cozmo's interactive abilities in the Star Wars universe. Check out the code and documentation for a Captivating Cozmo experience!

## Table of Contents

1. [Introduction](#introduction)
2. [Goals](#goals)
3. [Development Steps](#development-steps)
4. [Challenges](#challenges)
5. [Technical Details](#technical-details)
6. [Limitations](#limitations)
7. [Code Execution](#code-execution)

## Introduction

Welcome to Cozmo's Star Wars Odyssey! This project enriches Cozmo's interactive capabilities by incorporating computer vision for recognizing Star Wars characters. The README provides an overview of the project, development steps, challenges, and technical details.

## Goals

1. **CV Model Development:** Implement a model for recognizing Star Wars characters.
2. **ChatGPT Integration:** Enhance storytelling with dynamic responses.
3. **Programmatic User Queries:** Enable users to extend conversations.

## Development Steps

1. **Generating Training Data:** Captured 1000+ images per character in a controlled environment.
2. **Model Selection:** Implemented a CNN model with three layers for effective character recognition.
3. **Training Model:** Achieved 90.81% accuracy over 10 epochs.
4. **ChatGPT Integration:** Seamlessly integrated ChatGPT for context-aware conversations.
5. **Random Queries:** Developed diverse queries for engaging interactions.
6. **User Interface:** Created a simple interface for easy user interaction.

## Challenges

1. **Generating Training Data:** Time-consuming manual capture of images.
2. **Training Data with Different Backgrounds:** Addressed diversity by creating a consistent background.
3. **Selecting Model and Layers:** Explored architectures for optimal model selection.
4. **Parallel Execution:** Managed internet connectivity constraints for ChatGPT integration.

## Technical Details

- **Robot Interaction:** Cozmo library
- **Image Processing:** cv2 (OpenCV), PIL
- **Natural Language Processing:** OpenAI
- **Machine Learning:** TensorFlow, Keras
- **Data Handling:** NumPy, pandas

## Limitations

- Cozmo's camera resolution may impact precision.
- Limited dataset may affect adaptability.
- ChatGPT integration introduces cost considerations.
- Cozmo's limited battery life affects interaction duration.

## Code Execution

- In code to run this project, you will need a Cozmo Robot
- You will have to run the Python script in the Trainin_model folder to train the model
- Copy the saved model to the main_code folder and run predict.py 
