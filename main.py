############################################################################
# Text to speech
from gtts import gTTS
from pydub import AudioSegment

# working with audio
from scipy.io import wavfile

# local imports
import detect 
import torchs2t as speech_rec
import chat

# data processing
import cv2
import numpy as np
import av

# streamlit imports
import streamlit as st

from streamlit_webrtc import (
    AudioProcessorBase,
    ClientSettings,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)


# others imports
try:
    from typing import Literal, List
except ImportError:
    from typing_extensions import Literal, List  # type: ignore

import logging.handlers
import queue
from pathlib import Path
import asyncio
import time
import torch
############################### END of imports ###############################


# Initialization
HERE = Path(__file__).parent

logger = logging.getLogger(__name__)

WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": True},
)

WEBRTC_CLIENT_SETTINGS_video = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)

WEBRTC_CLIENT_SETTINGS_audio= ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": False, "audio": True},
)


st.title('A.I assistant on browser')

################# Object detection ##################################



def object_detection():
    class VideoProcessor(VideoProcessorBase):
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            xx1, yy1, xx2, yy2 = [0, 300, 0, 400] # inside of this box traffic light will get recognized.
            traffic_light_color = None
            car_state = None

            frame = frame.to_ndarray(format="bgr24")
            frame = cv2.resize(frame, (352, 288)) # chnage frame size -----------------------------(todo)

            frame_, car_state = detect.detect_car_movement(frame)

            frame, results = detect.object_detection(frame) 

            saved_in_dictionary = torch.load('saved_in_dictionary.pt')
            saved_in_dictionary['car_state'] = car_state
            saved_in_dictionary['detection_results'] = results

            for result in results:
                x1, y1, x2, y2, conf, class_ = result

                if class_ == 9: # this is a traffic light
                    if inside_box([xx1, yy1,xx2,yy2], [x1, y1, x2, y2]): # traffic light is inside of defined region
                        traffic_light_color = detect.img_classify(frame) # green, red, yellow
                        saved_in_dictionary['traffic_light_color'] = traffic_light_color

                        if traffic_light_color == 'red' and car_state == 'moving':
                            speak('stop the car, traffic light is red')
                            
                        if traffic_light_color == 'green' and car_state == 'stop':
                            speak('Traffic light turned to green')

            
            torch.save(saved_in_dictionary, 'saved_in_dictionary.pt')
            return av.VideoFrame.from_ndarray(frame, format="bgr24")


    webrtc_streamer(
        key="object_detection",
        mode=WebRtcMode.SENDRECV,
        client_settings=WEBRTC_CLIENT_SETTINGS_video,
        video_processor_factory=VideoProcessor,
        async_processing=True,
    )


def inside_box(big_box, small_box):
    x1 = small_box[0] - big_box[0]
    y1 = small_box[1] - big_box[1]
    x2 = big_box[2] - small_box[2]
    y2 = big_box[3] - small_box[3]
    return not bool(min([x1, y1, x2, y2, 0]))

#################################### Audio player ##################################

idxx=0
speak_bool = False
text_to_be_speak = ""

def get_audio_frame(n=900, audio_file='data.wav'): # get audio frame from a audio file 
    global idxx
    global speak_bool
    out = None

    if speak_bool:
        samplerate, data = wavfile.read(audio_file) # cant read generated audio file 
        out_list = [data[i:i+n].reshape(1, n) for i in range(0, len(data), n)[:-1]]

        if len(out_list)>idxx:
            out = out_list[idxx]  # data at this idxx number
        else:
            speak_bool = False
            idxx = 0

        idxx+=1
    return out, speak_bool


def audio_player():
    class AudioPlayer(AudioProcessorBase):
        def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
            raw_numpy, speak_bool = get_audio_frame()
            if speak_bool:
                new_audio = av.AudioFrame.from_ndarray(raw_numpy.astype('int16'), 's16', layout='stereo')
                new_audio.rate = 12000
                new_audio.pts = 0
                return new_audio
            else:
                speech_recognition(frame) 
                silence = av.AudioFrame.from_ndarray(np.zeros((1,1920)).astype('int16'), layout=frame.layout.name)
                silence.pts=0
                silence.rate=12000
                return silence


    webrtc_ctx = webrtc_streamer(
        key="audio_player",
        mode=WebRtcMode.SENDRECV,
        client_settings=WEBRTC_CLIENT_SETTINGS_audio,
        audio_processor_factory=AudioPlayer,
        async_processing=True,
    )


def speak(text):
    global speak_bool
    if text == '':
        text = 'could not hear you'
    gTTS(text).save('data.mp3')
    sound = AudioSegment.from_mp3("data.mp3")
    sound.export("data.wav", format="wav")
    speak_bool = True


########################################### End of audio player ############################


sample_s = np.array([]) 
listen_command = False
def speech_recognition(frame): # no need to call this function 
    global sample_s
    global listen_command
    frame_rate = 96000 # frame rate
    saved_audio_file = 'data.wav' 
    recording_duration = 500000  # 5 sec

    sample = frame.to_ndarray()[0].astype('int16')
    sample_s = np.concatenate((sample_s, sample)).astype('int16')
    if sample_s.shape[0] > recording_duration: # how long sentence to be record? : 5 sec?
        wavfile.write(saved_audio_file, frame_rate, sample_s) # saving audio file
        sentence = speech_rec.trigger_word_detect(saved_audio_file)
        # print('----->>>>', sentence)

        if listen_command: # activated at prev cycle
            listen_command = False
            reply = chat.chat_me(sentence) # chat bot
            if reply != None:
                speak(reply)

        elif sentence.find("hello") != -1: # if not listining yet, try to listen for trigger word from the sentence
            listen_command = True
            speak("listining")

        sample_s = np.array([])  # clearing array



# ################# another way to send data to browser #################
# def record_audio_frames():
#     webrtc_ctx = webrtc_streamer(
#         key="audio_recorder",
#         mode=WebRtcMode.SENDONLY,
#         audio_receiver_size=400,
#         client_settings=WEBRTC_CLIENT_SETTINGS_audio,
#         async_processing=True,
#     )
#     while True:
#         if webrtc_ctx.audio_receiver:
#             try:
#                 audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
#             except queue.Empty:
#                 logger.warning("Queue is empty. Abort.")
#                 break
#             for audio_frame in audio_frames:
#                 speech_recognition(audio_frame)
#         else:
#             logger.warning("AudioReciver is not set. Abort.")
#             break
#########################################################################

st.write('Object detection')
object_detection()


st.write('Audio')
audio_player()
