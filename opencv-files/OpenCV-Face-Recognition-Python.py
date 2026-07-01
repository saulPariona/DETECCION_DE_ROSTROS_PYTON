# coding: utf-8
import cv2
import os
import numpy as np

# ==========================================
# 1. CONFIGURACIÓN E INICIALIZACIÓN
# ==========================================
# Definimos las rutas basándonos en tu estructura de proyecto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(os.path.dirname(BASE_DIR), "training-data")
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Mapeo dinámico: el ID 0 se queda vacío como en el tutorial original
subjects = [""] 

# Inicializar detector y reconocedor LBPH
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# ==========================================
# 2. FUNCIONES DE DETECCIÓN Y DIBUJO
# ==========================================

def detect_face(img):
    """Detecta rostro, lo recorta y devuelve el área gris y el rectángulo."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=7)
    
    if len(faces) == 0:
        return None, None
    
    # Filtramos por tamaño para evitar falsos positivos
    for (x, y, w, h) in faces:
        if w > 50 and h > 50:
            return gray[y:y+h, x:x+w], (x, y, w, h)
            
    return None, None

def draw_rectangle(img, rect):
    """Dibuja un recuadro verde alrededor del rostro detectado."""
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

def draw_text(img, text, x, y):
    """Escribe el nombre del sujeto sobre el recuadro."""
    cv2.putText(img, text, (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

# ==========================================
# 3. PREPARACIÓN DE DATOS 
# ==========================================

def prepare_training_data(data_folder_path):
    dirs = os.listdir(data_folder_path)
    faces, labels = [], []
    
    for dir_name in dirs:
        # Asignamos nombre al sujeto y guardamos el índice
        subjects.append(dir_name)
        label = len(subjects) - 1
        
        subject_dir_path = os.path.join(data_folder_path, dir_name)
        for image_name in os.listdir(subject_dir_path):
            if image_name.startswith("."): continue
            
            image_path = os.path.join(subject_dir_path, image_name)
            image = cv2.imread(image_path)
            
            face, rect = detect_face(image)
            if face is not None:
                faces.append(face)
                labels.append(label)
    
    return faces, labels

# ==========================================
# 4. FLUJO PRINCIPAL
# ==========================================

print("Preparando datos...")
faces, labels = prepare_training_data(TRAIN_DIR)
print(f"Datos preparados. Sujetos: {subjects[1:]}")

# Entrenar el reconocedor
face_recognizer.train(faces, np.array(labels))
print("Entrenamiento completado.")

def predict(test_img):
    """Realiza la predicción sobre una imagen de prueba."""
    img = test_img.copy()
    face, rect = detect_face(img)
    
    if face is not None:
        label, confidence = face_recognizer.predict(face)
        label_text = subjects[label] if confidence < 100 else "Desconocido"
        
        draw_rectangle(img, rect)
        draw_text(img, label_text, rect[0], rect[1])
    return img

# Iniciar reconocimiento en video (Cámara)
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame = predict(frame)
    cv2.imshow("Reconocimiento Facial", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()