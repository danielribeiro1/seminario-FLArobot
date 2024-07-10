# Importações
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from joblib import load
from djitellopy import Tello
import time

# Constantes
LISTA_COMANDOS = ["parado/outros", "Esquerda", "Direita", "Se aproximar",
                  "Se afastar", "pousar", "trocar de drone", "subir", "descer"]
THRESHOLD = 0.6
DEFAULT_COMMAND = 2

# Desenha os landmarks do mediapipe
def draw_landmarks(image, landmarks):
    mp_draw.draw_landmarks(
        image,
        landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_draw.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
        connection_drawing_spec=mp_draw.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
    )

# Obtém os landmarks
def process_landmarks(landmarks):
    landmarks_data = {}
    landmark_names = [landmark.name for landmark in mp_pose.PoseLandmark]
    for i, landmark in enumerate(landmarks.landmark):
        if i not in list(range(0, 11)) + list(range(25, 33)):  # Exclui os índices indesejados.
            landmarks_data[f'{landmark_names[i]}.x'] = round(landmark.x, 4)
            landmarks_data[f'{landmark_names[i]}.y'] = round(landmark.y, 4)
            landmarks_data[f'{landmark_names[i]}.z'] = round(landmark.z, 4)
            landmarks_data[f'{landmark_names[i]}.visibility'] = round(landmark.visibility, 4)
    return pd.DataFrame([landmarks_data])

# Faz a predição da posição do operador a partir da MLP
def predict_command(model, data):
    probas = model.predict_proba(data)[0]
    pred_class = model.classes_[np.argmax(probas)]
    command_confidence = np.max(probas)
    if command_confidence < THRESHOLD:
        return "parado/outros", DEFAULT_COMMAND
    return LISTA_COMANDOS[pred_class], round(command_confidence, 2)

# Executa um comando com base na posição detectada
def execute_command(command, tello):
    commands = {
        "parado/outros": lambda: tello.send_rc_control(0, 0, 0, 0),
        "Esquerda": lambda: tello.send_rc_control(10, 0, 0, 0),
        "Direita": lambda: tello.send_rc_control(-10, 0, 0, 0),
        "Se aproximar": lambda: tello.send_rc_control(0, 10, 0, 0),
        "Se afastar": lambda: tello.send_rc_control(0, -10, 0, 0),
        "pousar": lambda: tello.land(),
        "trocar de drone": lambda: tello.land(),
        "subir": lambda: tello.send_rc_control(0, 0, 10, 0),
        "descer": lambda: tello.send_rc_control(0, 0, -10, 0),
    }
    if command in commands:
        commands[command]()

# Inicialização
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
drawing_styles = mp.solutions.drawing_styles
pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()

modelo_carregado = load('mlp_classifier.joblib')
    
# Inicializa os movimentos do drone
tello = Tello()
tello.connect()
tello.streamon() # Inicializa a câmera do drone
tello.takeoff() # Inicializa o voo do drone
time.sleep(2.4)
tello.send_rc_control(0, 0, 55, 0) # Define uma altura para o drone inicializar a captura de movimentos
time.sleep(2.4)


with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:

    tello.send_rc_control(0, 0, 0, 0) # Encerra qualquer movimento do drone para entrar no loop principal

    # Loop principal
    while True:
        img_bgr = tello.get_frame_read().frame
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results_pose = pose.process(img_rgb)

        if results_pose.pose_landmarks:
            draw_landmarks(img_rgb, results_pose.pose_landmarks)
            new_row = process_landmarks(results_pose.pose_landmarks)
            command, confidence = predict_command(modelo_carregado, new_row)
            result_text = f"{command} - {confidence}"
            cv2.putText(img_rgb, result_text, (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            execute_command(command, tello)

        cv2.putText(img_rgb, "Pressione 'Q' para sair.", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow('MediaPipe Pose', img_rgb)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
