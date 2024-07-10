import time, sys, os, cv2
import pandas as pd
import mediapipe as mp

df_landmarks = pd.DataFrame()
cwd = os.getcwd()
images_directory = 'data/'
images_directory = os.path.join(cwd, images_directory)
csv_file_path = 'datapose.csv'
image_counter = len(os.listdir(images_directory))

# Inicializa o MediaPipe Pose.
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

# Configurações de desenho para a pose.
drawing_styles = mp.solutions.drawing_styles
pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()

# Inicializa a captura de vídeo.
cap = cv2.VideoCapture(0)

with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        ret, image = cap.read()
    
        # Verifica se o frame foi capturado corretamente
        if not ret:
            print("Erro ao capturar o frame; saindo...")
            sys.exit(0)

        # Espelha a imagem para evitar efeito de imagem invertida.
        image = cv2.flip(image, 1)

        # Converte a imagem de BGR para RGB.
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Processa a imagem e detecta a pose.
        results_pose = pose.process(image_rgb)

        # Desenha as marcações da pose se uma pose for detectada.
        if results_pose.pose_landmarks:
            mp_draw.draw_landmarks(
                image,
                results_pose.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_draw.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                connection_drawing_spec=mp_draw.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
            landmarks_data = {}
            landmark_names = [landmark.name for landmark in mp_pose.PoseLandmark]
            for i, landmark in enumerate(results_pose.pose_landmarks.landmark):
                if i not in list(range(0, 11)) + list(range(25, 33)):  # Exclui os índices que você não quer.
                    landmarks_data[f'{landmark_names[i]}.x'] = round(landmark.x, 4)
                    landmarks_data[f'{landmark_names[i]}.y'] = round(landmark.y, 4)
                    landmarks_data[f'{landmark_names[i]}.z'] = round(landmark.z, 4)
                    landmarks_data[f'{landmark_names[i]}.visibility'] = round(landmark.visibility, 4)

            # Cria um DataFrame com uma única linha usando o dicionário acima.
            new_row = pd.DataFrame([landmarks_data])
            # Adiciona a coluna 'Label' com valor padrão 1.
            new_row['Label'] = 1
            image_path = f'{images_directory}image_{image_counter}.png'
            new_row['Image_Path'] = image_path 
            cv2.imwrite(image_path, image)

            # Concatena a nova linha ao DataFrame existente.
            df_landmarks = pd.concat([df_landmarks, new_row], ignore_index=True)

            df_landmarks['Label'] = 1
            new_row.to_csv(csv_file_path, mode='a', header=False, index=False)
            print(df_landmarks)

            image_counter += 1
            time.sleep(1)

        # Instruções na janela.
        cv2.putText(image, "Pressione 'Q' para sair.", (5, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

        # Exibe a imagem resultante.
        cv2.imshow('MediaPipe Pose', image)

        # Fecha a janela ao pressionar a tecla 'q'.
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Libera a captura e fecha as janelas abertas.
cap.release()
cv2.destroyAllWindows()
