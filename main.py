import cv2
import dlib
import numpy

PREDICTOR_PATH = "./data/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)

cascade_path='./haarcascade_frontalface_alt2.xml'
cascade = cv2.CascadeClassifier(cascade_path)

def get_landmarks(im):
    rects = cascade.detectMultiScale(im, 1.3,5)
    (x,y,w,h) = rects[0]
    rect = dlib.rectangle(x,y,x+w,y+h)
    return numpy.matrix([[p.x, p.y] for p in predictor(im, rect).parts()])

def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
            fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
            fontScale=0.4,
            color=(255, 0, 0))
        cv2.circle(im, pos, 2, color=(255, 255, 0))
    return im

if __name__ == "__main__" :
    im=cv2.imread('Lenna.jpg')
    cv2.imshow('result',annotate_landmarks(im,get_landmarks(im)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

