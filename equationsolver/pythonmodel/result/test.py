import numpy as np
import tensorflow as tf
import cv2
from os.path import isfile, join
from PIL import Image
import image


def convert():
    # image = Image.open("D:/Python Projects/Python Practice/result/Untitled.png")
    # image = image.resize((28,28), Image.LANCZOS) 
    # image.save("Untitled.png",quality=95)
    # image = cv2.imread("D:/Python Projects/Python Practice/result/Untitled.png")
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    # image = np.reshape(image, (28, 28, -1))
    # image = np.reshape(image, (28, 28, 1)).astype('float32')
    # image = np.expand_dims(image, axis=0)
    equation = (image.processor("D:/Python Projects/Python Practice/result/Untitled.png"))
    print('\n equation :', equation)
    #return image

def getResult(output_data):
    mapList = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', '-',
               '+', 'w', 'x', 'y', 'z','slash', 'equals']
    output_data = output_data[0].tolist()
    return mapList[output_data.index(1.0)]

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="models.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("== Input details ==")
print("shape:", input_details[0]['shape'])
print("type:", input_details[0]['dtype'])
print("\n== Output details ==")
print("shape:", output_details[0]['shape'])
print("type:", output_details[0]['dtype'])

img_float32 = convert()

# Test model on random input data.
# input_shape = input_details[0]['shape']
# input_data = img_float32
# interpreter.set_tensor(input_details[0]['index'], input_data)

# interpreter.invoke()

# # The function `get_tensor()` returns a copy of the tensor data.
# # Use `tensor()` in order to get a pointer to the tensor.
# output_data = interpreter.get_tensor(output_details[0]['index'])
# print(">>>>>")
# print (getResult(output_data))