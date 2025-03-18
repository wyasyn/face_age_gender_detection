import cv2
import numpy as np

# Load pre-trained models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

face_model_path = "./models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
face_proto_path = "./models/deploy.prototxt"

age_model_path = "./models/age_net.caffemodel"
age_proto_path = "./models/age_deploy.prototxt"

gender_model_path = "./models/gender_net.caffemodel"
gender_proto_path = "./models/gender_deploy.prototxt"

age_net = cv2.dnn.readNetFromCaffe(age_proto_path, age_model_path)
gender_net = cv2.dnn.readNetFromCaffe(gender_proto_path, gender_model_path)
face_net = cv2.dnn.readNetFromCaffe(face_proto_path, face_model_path)

AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
GENDER_BUCKETS = ["Male", "Female"]


# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
        
        # Predict gender
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = GENDER_BUCKETS[gender_preds[0].argmax()]
        
        # Predict age
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = AGE_BUCKETS[age_preds[0].argmax()]
        
        # Draw face rectangle and labels
        label = f"{gender}, {age}"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    cv2.imshow("Age and Gender Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
