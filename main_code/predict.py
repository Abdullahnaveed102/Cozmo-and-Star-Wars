#import the cozmo and image libraries
import os
import cv2
import time
import cozmo
import _thread
import asyncio
from PIL import Image
from cozmo.util import degrees, distance_mm
from cozmo.objects import LightCube1Id, LightCube2Id, LightCube3Id

import openai
import random
import numpy as np
import pandas as pd
from tensorflow import keras
from keras.optimizers import Adam
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array

#=========================================================================================================>
#==========================================================================================> Global Things
desired_head_angle = 5
loaded_model = load_model('cozmo_model.h5')
class_mapping = {0: 'C-3PO', 1: 'Ewoks', 2: 'Finn', 3: 'R2-D2', 4: 'Vader'}
path_to_image_folder = 'captured_imgs/'
image_name = 0

openai.api_key = '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'
messages = [{"role": "system", "content": "You are a kind helpful assistant."}]

#=========================================================================================================>
#=====================================================================================> Make Query for GPT
def make_query_and_ask(character):
    q1 = "A fun fact about " + character + " (1 sentence only)."
    q2 = "Who played the part of " + character + " (1 sentence only)."
    q3 = "How does " + character + " personality evolve or change throughout the Star Wars movies? (1 sentence only)."
    q4 = character + "'s role key contribution in the Star Wars series? (1 sentence only)."
    q5 = "In which Star Wars movies does " + character + " play a significant or crucial role in the plot? (1 sentence only)."

    queries = [q1, q2, q3, q4, q5]
    message = random.choice(queries)

    if message:
        messages.append({"role": "user", "content": message},)
        chat = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages,)
    
    reply = chat.choices[0].message.content
    messages.append({"role": "assistant", "content": reply})
    return reply

#=========================================================================================================>
#======================================================================================> predict_character
def predict_character(robot):
    global image_name
    img_path = path_to_image_folder + str(image_name) + '.png'

    img = image.load_img(img_path, target_size=(320, 240))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    predictions = loaded_model.predict(img_array)
    
    predicted_class = np.argmax(predictions[0])
    predicted_class_label = class_mapping[predicted_class]
    print(f'Predicted class: {predicted_class_label}')

    about = make_query_and_ask(predicted_class_label)
    about = predicted_class_label + ". " + about
    robot.say_text(about, duration_scalar=0.7).wait_for_completed()

    image_name = image_name + 1

#=========================================================================================================>
#==================================================================================> Take a Picture & Save
def capture_image(robot):
    global image_name

    number = image_name
    path_plus_name = path_to_image_folder + str(number)
    save_path = f"{path_plus_name}.png"
    image = robot.world.latest_image.raw_image
    image.save(save_path, 'PNG')

#=========================================================================================================>
#=================================================================================> Input from User helper
def input_from_user_helper(robot, message):
    if message:
        messages.append({"role": "user", "content": message},)
        chat = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages,)
    
    reply = chat.choices[0].message.content
    messages.append({"role": "assistant", "content": reply})
    return reply

#=========================================================================================================>
#========================================================================================> Input from User
def input_from_user(robot):
    file_path = 'user_input.txt'

    with open(file_path, 'r') as file:
        file_content = file.read()
    file_content = str(file_content)
    message = str(file_content)
    message = message.replace('\n', '') + " (1 sentence or one line only)."

    print(message)
    reply = input_from_user_helper(robot, message)

    robot.say_text(reply, duration_scalar=0.7).wait_for_completed()

#=========================================================================================================>
#=======================================================================================> has_file_updated
def has_file_updated(file_path, last_check_time):
    try:
        current_modification_time = os.path.getmtime(file_path)
        if current_modification_time > last_check_time:
            return True
        else:
            return False
    except FileNotFoundError:
        return False
    
#=========================================================================================================>
#==========================================================================================> cozmo_program
def cozmo_program(robot: cozmo.robot.Robot):
    robot.camera.color_image_enabled = True
    #=========================================>  Connect to cubes
    robot.world.connect_to_cubes()
 
    #=========================================> Identify cubes
    cube1 = robot.world.get_light_cube(LightCube1Id)  

    #=========================================> Testing
    if cube1 is not None:
        cube1.set_lights(cozmo.lights.green_light)
    else:
        cozmo.logger.warning("Cozmo is not connected to a LightCube1Id cube - check the battery.")
	
    #=========================================> Function to handle cube taps
    def on_cube_tap1(evt, **kwargs):
        # robot.set_head_angle(degrees(desired_head_angle))
        capture_image(robot)
        predict_character(robot)

    #=========================================> Add event handlers for cube taps
    cube1.add_event_handler(cozmo.objects.EvtObjectTapped, on_cube_tap1)

    file_path = "user_input.txt"
    last_check_time = os.path.getmtime(file_path)
    #=========================================> Keep the program running until the user stops it
    while True:
        if has_file_updated(file_path, last_check_time):
            print("File has been updated.")
            input_from_user(robot)
            last_check_time = time.time()
        time.sleep(1)

#=========================================================================================================>
#============================================================================================> run_program
cozmo.run_program(cozmo_program, use_viewer=True, force_viewer_on_top=True)