import numpy as np
import cv2
from keras.preprocessing.image import img_to_array
from keras.models import load_model

cap = cv2.VideoCapture(0)
shape = 28
model = load_model('lenet.model')

while 1:
    _, original = cap.read()
    ret, image = cap.read()
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = image.astype("float") / 255.0
    reshaped = cv2.resize(image, (shape, shape))
    image = img_to_array(reshaped)
    image = np.expand_dims(image, axis=0)

	# classify the input image
    (notSanta, santa) = model.predict(image)[0]

	# build the label
    label = "Ring" if santa > notSanta else "Not Ring"
    proba = santa if santa > notSanta else notSanta

    label1 = "Not Ring" if santa > notSanta else "Ring"
    proba1 = notSanta if santa > notSanta else santa

    label = "{}: {:.2f}%".format(label, proba * 100)
    print(label)
    
    cv2.imshow('img',reshaped)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()