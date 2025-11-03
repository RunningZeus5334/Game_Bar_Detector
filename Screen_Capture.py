import numpy as np
import cv2
from mss import mss
from PIL import Image

bounding_box = {'top': 100, 'left': 0, 'width': 1920, 'height': 1080}

sct = mss()

# Kies monitor 1 (kan 2 zijn als jouw hoofdscherm daar staat)
monitor = sct.monitors[1]

# Of maak een eigen bounding box gebaseerd op monitor 1:
bounding_box = {
    'top': monitor['top'],
    'left': monitor['left'],
    'width': monitor['width'],
    'height': monitor['height']
}

while True:
    sct_img = sct.grab(bounding_box)
    cv2.imshow('screen', np.array(sct_img))

    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()
        break

