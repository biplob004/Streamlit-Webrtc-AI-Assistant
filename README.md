# Streamlit-webrtc-ai-assistant
This is a simple ai assistant for car drivers to tell about traffic light and other stuffs. You can deploy this project on cloud and then can access from web browser.  Download resnet50 image classifictaion model from this link and place it on this the main folder https://drive.google.com/file/d/15h92yacbwYQ0c1YeR63PSpKOPnQ5klu6/view?usp=sharing



# List of features availabe in this project

1. Object detection using yolov5s.pt weight file, (for detection of traffic light and other objects)
2. Image classification (resnet50) : for classifiction of traffic lights into green, red and yollow colors.
3. Voice recogniton : Speech is converting into text using silero pytorch speech recognition model.
4. Text to speech: Using google gTTS, text is converting into audio file and that audio file is playing for voice reply.
5. Chat bot is using from github repo (https://github.com/python-engineer/pytorch-chatbot) 
6. Using opencv-python a function that detects wheather car is moving or not.
7. Streamlit webrtc: so, we can run this project on our phone after deploing on aws or on any other cloud server.

# What this project actually does?
1. It is watching traffic light all the time from phone camera, and if car is moving while traffic ligh is red then it will wanrn the driver by voice message.
2. It have a chat bot, so you can talk to the ai assistant. (customizable chatbot)
3. You can ask the ai assistant what he see on his camera, then he will reply.

