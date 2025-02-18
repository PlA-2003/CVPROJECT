import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import argparse

# Load TensorFlow EmotionCNN model
emotion_model = load_model(r'.\models\emotion1.h5')

# List of emotions
emotion_list = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Neutral', 'Happy']

# Load các mô hình OpenCV
faceProto = "./models/opencv_face_detector.pbtxt"
faceModel = "./models/opencv_face_detector_uint8.pb"

# Load các mô hình Age và Gender
ageProto = "./models/age_deploy.prototxt"
ageModel = "./models/age_net.caffemodel"
genderProto = "./models/gender_deploy.prototxt"
genderModel = "./models/gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Load face detection, age, and gender models
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Define highlightFace function for face detection
def highlightFace(faceNet, frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177, 123), swapRB=False)
    faceNet.setInput(blob)
    detections = faceNet.forward()
    faceBoxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            faceBoxes.append((x1, y1, x2, y2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frame, faceBoxes

# Transformation for emotion input
# Chuyển đổi hình ảnh thành đầu vào cho mô hình cảm xúc
# Chuyển đổi hình ảnh thành đầu vào cho mô hình cảm xúc
def preprocess_emotion_input(face_img):
    face_img = Image.fromarray(face_img)
    face_img = face_img.resize((48, 48))  # Thay đổi kích thước thành 48x48
    face_img = np.array(face_img)  # Chuyển sang mảng numpy

    if face_img.shape[-1] == 1:  # Nếu ảnh là grayscale, chuyển thành RGB
        face_img = np.repeat(face_img, 3, axis=-1)
    
    face_img = face_img / 255.0  # Chuẩn hóa ảnh (0-1)
    face_img = np.expand_dims(face_img, axis=0)  # Thêm chiều batch
    return face_img


# Setup video input
parser = argparse.ArgumentParser()
parser.add_argument('--image', help="Path to image or 0 for webcam")
args = parser.parse_args()

video = cv2.VideoCapture(args.image if args.image else 0)
padding = 20

while cv2.waitKey(1) < 0:
    hasFrame, frame = video.read()
    if not hasFrame:
        print("No frame captured, closing video stream.")
        break

    resultImg, faceBoxes = highlightFace(faceNet, frame)
    
    if not faceBoxes:
        print("No face detected")
        continue

    for faceBox in faceBoxes:
        face = frame[max(0, faceBox[1] - padding):min(faceBox[3] + padding, frame.shape[0] - 1),
                     max(0, faceBox[0] - padding):min(faceBox[2] + padding, frame.shape[1] - 1)]

        if face.size == 0:
            continue

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        print(f'Gender: {gender}')

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        print(f'Age: {age[1:-1]} years')

        emotion_input = preprocess_emotion_input(face)
        emotion_preds = emotion_model.predict(emotion_input)
        emotion = np.argmax(emotion_preds, axis=1)[0]
        emotion_label = emotion_list[emotion]
        print(f'Emotion: {emotion_label}')

        cv2.putText(resultImg, f'{gender}, {age}, {emotion_label}', (faceBox[0], faceBox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Detecting age, gender, and emotion", resultImg)

video.release()
cv2.destroyAllWindows()
