import numpy as np
import tensorflow as tf
import cv2
import segmentor
from os.path import isfile, join


def convert():
    segmentor.image_segmentation("D:/Python Projects/Python Practice/result/Untitled2.png")
    image = cv2.imread("D:/Python Projects/Python Practice/result/1_1_2.png")
    image = cv2.resize(image, (28, 28), interpolation = cv2.INTER_CUBIC)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    image = np.reshape(image, (28, 28, -1))
    image = np.reshape(image, (28, 28, 1)).astype('float32')
    image = np.expand_dims(image, axis=0)
    return image

def getResult(output_data):
    print(output_data);
    mapList = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', '-',
               '+', 'w', 'x', 'y', 'z','slash', 'equals']
    output_data = output_data[0].tolist()
    return mapList[output_data.index(1.0)]

interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("== Input details ==")
print("shape:", input_details[0]['shape'])
print("type:", input_details[0]['dtype'])
print("\n== Output details ==")
print("shape:", output_details[0]['shape'])
print("type:", output_details[0]['dtype'])

img_float32 = convert()

input_shape = input_details[0]['shape']
input_data = img_float32
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])
print(">>>>>")
print (getResult(output_data))