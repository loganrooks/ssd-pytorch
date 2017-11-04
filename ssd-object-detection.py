#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable
import cv2
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio

def detectObjects(originalImage, neuralNet, transform):
    height, width = originalImage.shape[:2]
    transformedImage = transform(originalImage)[0]
    imageTensor = Variable(torch.from_numpy(transformedImage).permute(2, 0, 1).unsqueeze(0))
    detections = neuralNet(imageTensor).data
    # detection = (batch #, # of classes, # of class occurence, (score, x0, y0, x1, y1))
    scaleTensor = torch.Tensor([width, height, width, height])
    numOfClasses = detections.size(1)
    
    for i in range(numOfClasses):
        j = 0
        currentClass = labelmap[i - 1]
        score = detections[0, i, :, 0]
        while score[j] >= 0.6:
            
            objectCoordinates = (detections[0, i, j, 1:] * scaleTensor).numpy()
            rectangleColour = (255, 0, 0)
            rectangleThickness = 2
            cv2.rectangle(originalImage, (objectCoordinates[0], objectCoordinates[1]), 
                                          (objectCoordinates[2], objectCoordinates[3]), 
                          rectangleColour, rectangleThickness)
            
            (textColour, textFont, textSize) = (255, 255, 255), cv2.FONT_HERSHEY_SIMPLEX, 2
            cv2.putText(originalImage, currentClass, (objectCoordinates[0], objectCoordinates[1]), 
                        textFont, textSize, textColour, textSize, cv2.LINE_AA)
            
            j += 1
            
    return originalImage

if __name__ == "__main__":
    neuralNet = build_ssd('test')
    neuralNet.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth', map_location = lambda storage, loc: storage))
    trainScale = (104/256.0, 117/256.0, 123/256.0)
    transform = BaseTransform(neuralNet.size, trainScale)
    
    videoReader = imageio.get_reader('funny_dog.mp4')
    fps = videoReader.get_meta_data()['fps']
    videoWriter = imageio.get_writer('output.mp4', fps = fps)
    for i, frame in enumerate(videoReader):
        processedFrame = detectObjects(frame, neuralNet.eval(), transform)
        videoWriter.append_data(processedFrame)
        print(i)
    videoWriter.close()
        