import os
import torch
import torchvision
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import base64
from PIL import Image
from io import BytesIO
from tensorflow.keras.models import load_model
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# Cấu hình Flask và các thư viện
app = Flask(__name__)

# Định nghĩa thư mục tải lên
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Tải mô hình Faster R-CNN
def load_fasterrcnn_model(model_path):
    # Tải mô hình Faster R-CNN có sẵn từ torchvision
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    
    # Nạp trọng số vào mô hình
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Chế độ inference
    return model

# Đường dẫn tới mô hình Faster R-CNN
model_path = './models/fastercnn.pth'  # Đảm bảo rằng đường dẫn này là đúng
model = load_fasterrcnn_model(model_path)

# Load mô hình EmotionCNN
emotion_model = load_model(r'./models/emotion1.h5')  # Đường dẫn tới mô hình EmotionCNN của bạn

# Load các mô hình AgeNet và GenderNet
ageProto = "./models/age_deploy.prototxt"
ageModel = "./models/age_net.caffemodel"
genderProto = "./models/gender_deploy.prototxt"
genderModel = "./models/gender_net.caffemodel"
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Các danh sách cho giới tính và độ tuổi
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']
emotionList = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Neutral', 'Happy']

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# Kiểm tra định dạng file hợp lệ
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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

# Nhận diện đối tượng với Faster R-CNN
def detect_objects(frame, model, threshold=0.5):
    # Chuyển đổi hình ảnh thành tensor
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    image_tensor = transform(frame).unsqueeze(0)

    # Dự đoán với mô hình Faster R-CNN
    with torch.no_grad():
        prediction = model(image_tensor)

    boxes = prediction[0]['boxes'].cpu().numpy()  # Lấy bounding boxes
    labels = prediction[0]['labels'].cpu().numpy()  # Lấy labels
    scores = prediction[0]['scores'].cpu().numpy()  # Lấy scores

    # Chỉ giữ các đối tượng có score > threshold và label = 1 (ví dụ khuôn mặt)
    filtered_boxes = []
    for i in range(len(scores)):
        if scores[i] > threshold and labels[i] == 1:  # Giữ lại đối tượng với label = 1 (ví dụ khuôn mặt)
            filtered_boxes.append(boxes[i])

    return filtered_boxes

# Nhận diện cảm xúc, giới tính và độ tuổi
def detect_emotion_age_gender(frame, model):
    faceBoxes = detect_objects(frame, model)  # Sử dụng mô hình Faster R-CNN
    results = []

    for faceBox in faceBoxes:
        x1, y1, x2, y2 = faceBox
        face = frame[int(y1):int(y2), int(x1):int(x2)]

        # Dự đoán cảm xúc
        emotion_input = preprocess_emotion_input(face)
        emotion_preds = emotion_model.predict(emotion_input)
        emotion = np.argmax(emotion_preds, axis=1)[0]
        emotion_label = emotionList[emotion]

        # Dự đoán giới tính và độ tuổi
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]  # Lấy giới tính với xác suất cao nhất

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]  # Lấy độ tuổi với xác suất cao nhất

        results.append({
            "gender": gender,
            "age": age,
            "emotion": emotion_label,
            "face_box": (x1, y1, x2, y2)
        })

    return results

@app.route('/')
def index():
    return render_template('index.html')

# API nhận diện đối tượng từ ảnh tải lên
@app.route('/detect', methods=['POST'])
def detect_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Mở và xử lý ảnh
        frame = cv2.imread(filepath)
        results = detect_emotion_age_gender(frame, model)

        # Vẽ bounding box cho các đối tượng
        for result in results:
            x1, y1, x2, y2 = result["face_box"]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{result['emotion']}, {result['gender']}, {result['age']}", 
                        (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Chuyển đổi ảnh thành base64 để hiển thị trên web
        _, img_encoded = cv2.imencode('.jpg', frame)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')

        return jsonify({"imageUrl": "data:image/jpeg;base64," + img_base64})

# API nhận diện khuôn mặt từ webcam frame
@app.route('/detect_frame', methods=['POST'])
def detect_frame():
    data = request.get_json()
    img_data = base64.b64decode(data['image'].split(',')[1])
    image = Image.open(BytesIO(img_data))
    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    results = detect_emotion_age_gender(frame, model)

    # Vẽ bounding box cho khuôn mặt
    # Vẽ bounding box cho các đối tượng
    for result in results:
        x1, y1, x2, y2 = result["face_box"]

        # Đảm bảo các tọa độ là số nguyên
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Vẽ bounding box trên ảnh
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{result['emotion']}, {result['gender']}, {result['age']}", 
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


    # Chuyển đổi ảnh thành base64 để hiển thị trên web
    _, img_encoded = cv2.imencode('.jpg', frame)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')

    return jsonify({"imageUrl": "data:image/jpeg;base64," + img_base64})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
