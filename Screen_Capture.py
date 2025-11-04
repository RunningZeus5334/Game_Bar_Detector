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
    # Grabbing screen data
    sct_img = sct.grab(bounding_box)

    # Convert to PIL Image
    frame = np.array(sct_img)

    # Converteer naar grijs
    gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)

    # Canny edge detection toepassen
    edges = cv2.Canny(gray, 50, 100,) 

    # Converting to NumPy array and displaying
    cv2.imshow('screen', np.array(sct_img))
    
    # Toon het resultaat
    cv2.imshow('Edges', edges)





    # Stoppen met 'q'
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()
        break

