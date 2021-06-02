import time
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from models.experimental import attempt_load
from utils1.general import non_max_suppression
from torchvision import models
import random
import math
from torchvision import transforms
import time
import requests
from PIL import Image


yolov5_weight_file = 'yolov5s.pt' 
conf_set = 0.30 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = attempt_load(yolov5_weight_file, map_location=device)
cudnn.benchmark = True 
names = model.module.names if hasattr(model, 'module') else model.names


# saving names in dictionary file
torch.save({'traffic_light_color':'not detected', 'car_state':'none', 'detection_results':'nothing detected'}, 'saved_in_dictionary.pt') # new empty file

saved_in_dictionary = torch.load('saved_in_dictionary.pt')
saved_in_dictionary['yolo_names'] = names
torch.save(saved_in_dictionary, 'saved_in_dictionary.pt')



def object_detection(frame):
    img = torch.from_numpy(frame)
    img = img.permute(2, 0, 1 ).float().to(device)
    img /= 255.0  
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img, augment=False)[0]
    pred = non_max_suppression(pred, conf_set, 0.45) # prediction, conf, iou

    detection_result = []
    for i, det in enumerate(pred):
        if len(det): 
            for d in det: # d = (x1, y1, x2, y2, conf, cls)
                x1 = int(d[0].item())
                y1 = int(d[1].item())
                x2 = int(d[2].item())
                y2 = int(d[3].item())
                conf = round(d[4].item(), 2)
                c = int(d[5].item())
                
                detected_name = names[c]
                detection_result.append([x1, y1, x2, y2, conf, c])
                
                ## Bounding box
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2) # box
                frame = cv2.putText(frame, f'{names[c]} {str(conf)}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
    return (frame, detection_result)


############### Image classifier #####################


classification_weight_file = 'resnet50'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model2 = models.resnet50()
model2.fc = torch.nn.Linear(in_features=2048, out_features=3, bias=True)

model2.load_state_dict(torch.load(classification_weight_file, map_location=device))
model2 = model2.to(device)
model2.eval()


transform = transforms.Compose([
            transforms.Resize((224, 224)), 
            # transforms.CenterCrop(142),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                            [0.229, 0.224, 0.225])
          ]) 


def img_classify(frame):
    labels = ['green', 'red', 'yellow']
    frame = transform(Image.fromarray(frame))
    frame = frame.unsqueeze(0)
    prediction = model2(frame)
    result_idx = torch.argmax(prediction).item()
    return labels[result_idx]







################### motion of car ################
import cv2 as cv
import cv2
import numpy as np


backSub = cv.createBackgroundSubtractorMOG2()

def get_mask(image):
    ysize = image.shape[0]
    xsize = image.shape[1]
    color_select = np.copy(image)
    line_image = np.copy(image)

    # print(ysize, xsize)  # 720- 1280

    # Define our color criteria
    red_threshold = 0
    green_threshold = 0
    blue_threshold = 0
    rgb_threshold = [red_threshold, green_threshold, blue_threshold]

   
    left_bottom = [-800, ysize - 100]
    right_bottom = [xsize + 800, ysize - 100]
    apex = [xsize / 2, ysize / 3]

    fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
    fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
    fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

    # Mask pixels below the threshold
    color_thresholds = (image[:, :, 0] < rgb_threshold[0]) | \
                       (image[:, :, 1] < rgb_threshold[1]) | \
                       (image[:, :, 2] < rgb_threshold[2])

    # Find the region inside the lines
    XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
    region_thresholds = (YY > (XX * fit_left[0] + fit_left[1])) & \
                        (YY > (XX * fit_right[0] + fit_right[1])) & \
                        (YY < (XX * fit_bottom[0] + fit_bottom[1]))
    # Mask color selection
    color_select[color_thresholds] = [0, 0, 0]
    # Find where image is both colored right and in the region
    line_image[~color_thresholds & region_thresholds] = [255, 0, 0]

    return line_image


def split_bg_fg(seg):
    ## create fg/bg mask
    seg_gray = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY)
    _, bg_mask = cv2.threshold(seg_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
   
    bg = cv2.bitwise_and(seg, seg, mask=bg_mask)
    return bg


def detect_car_movement(frame):
    bg1 = get_mask(frame)

    bg = split_bg_fg(bg1)

    frame = bg

    fgMask = backSub.apply(bg)

    # print(fgMask.mean(), bg.mean())
    state = ""
    if fgMask.mean() > 20:
        state = 'moving'
    else:
        state = 'stop'
    # print(state)
    # print(frame.shape)

    cv.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
    cv.putText(frame, state, (0, 0),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 500)
    fontScale = 1
    fontColor = (255, 0, 255)
    lineType = 2

    cv2.putText(frame, state,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)


    # cv.imshow('Frame', frame)
    # cv.imshow('FG Mask', fgMask)

    # keyboard = cv.waitKey(30)
    # if keyboard == 'q' or keyboard == 27:
    #     break

    return [frame, state]












