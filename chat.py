import random
import json

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()




def chat_me(text):
    text = tokenize(text)
    X = bag_of_words(text, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    reply = ''
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                reply = random.choice(intent['responses'])
    else:
        reply = "not sure, what you saying"


    # accessinging all variables
    saved_in_dictionary = torch.load('saved_in_dictionary.pt')
    car_state = saved_in_dictionary['car_state']
    yolo_results = saved_in_dictionary['detection_results']
    traffic_light_color = saved_in_dictionary['traffic_light_color']

    if reply == "traffic_light_color":
        return f"traffic light color is {traffic_light_color}"

    elif reply == "surrounding":
        objects = " "
        for result in yolo_results:
            objects += saved_in_dictionary['yolo_names'][result[5]]
        return f" I can see {objects}"


    ##### more codes here ####

    return reply


