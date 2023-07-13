import numpy as np
import cv2
import os
import datetime
from tqdm import tqdm

'''
Este programa debiera corresponder a videos largos. Podrían ser tomados por camara web en tiempo real o un video al que 
se tome como si fuera una cámara, y se vayan perdiendo partes del video al procesar fragmentos de 5 segundos en serie. 
'''


class Ctes:
    DURATION_VIDEO_CUTS = 5
    FPS = 30


def filename_date():
    # Obtener la fecha y hora actual
    date_hour = datetime.datetime.now()

    # Formatear la fecha y hora actual en "ano_mes_dia_hora_minutos_segundos"
    formato = "%Y-%m-%d_%H-%M-%S"
    formatted_date_hour = date_hour.strftime(formato)

    return formatted_date_hour


def getOpticalFlow(video):
    """Calculate dense optical flow of input video
    Args:
        video: the input video with shape of [frames,height,width,channel]. dtype=np.array
    Returns:
        flows_x: the optical flow at x-axis, with the shape of [frames,height,width,channel]
        flows_y: the optical flow at y-axis, with the shape of [frames,height,width,channel]
    """
    # initialize the list of optical flows
    gray_video = []

    for i in range(len(video)):
        img = cv2.cvtColor(video[i], cv2.COLOR_RGB2GRAY)
        gray_video.append(np.reshape(img, (224, 224, 1)))

    flows = []
    for i in range(0, len(video) - 1):
        # calculate optical flow between each pair of frames
        flow = cv2.calcOpticalFlowFarneback(gray_video[i], gray_video[i + 1], None, 0.5, 3, 15, 3, 5, 1.2,
                                            cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        # subtract the mean in order to eliminate the movement of camera
        flow[..., 0] -= np.mean(flow[..., 0])
        flow[..., 1] -= np.mean(flow[..., 1])
        # normalize each component in optical flow
        flow[..., 0] = cv2.normalize(flow[..., 0], None, 0, 255, cv2.NORM_MINMAX)
        flow[..., 1] = cv2.normalize(flow[..., 1], None, 0, 255, cv2.NORM_MINMAX)
        # Add into list
        flows.append(flow)

    # Padding the last frame as empty array
    flows.append(np.zeros((224, 224, 2)))

    return np.array(flows, dtype=np.float32)


def get_camera_fps():
    # Open the camera
    cap = cv2.VideoCapture(0)
    # Obtener la velocidad de cuadros por segundo
    fps = cap.get(cv2.CAP_PROP_FPS) or Ctes.FPS
    cap.release()
    return fps


def procesar_videos(directory_path, fps):
    fragmento_duracion = 5  # Duración deseada de cada fragmento en segundos
    fragmento_frames = int(fragmento_duracion * fps)  # Número de frames por fragmento

    for filename in os.listdir(directory_path):
        if filename.endswith(".avi"):
            video_file = os.path.join(directory_path, filename)
            cap = cv2.VideoCapture(video_file)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            current_frame = 0

            while current_frame + fragmento_frames <= total_frames:
                # Leer el fragmento de video
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                ret, frame = cap.read()
                if not ret:
                    break
                frame_path = os.path.join(video_file + '_' + str(current_frame) + '.avi')
                # Procesar el fragmento de video
                Video2Npy(frame_path)

                # Avanzar al siguiente fragmento
                current_frame += fragmento_frames

            cap.release()
            print(f"Video {filename} procesado.")

        else:
            print(f"Ignorando el archivo {filename}.")


def Video2Npy(file_path, resize=(224, 224)):
    """Load video and tansfer it into .npy format
    Args:
        file_path: the path of video file
        resize: the target resolution of output video
    Returns:
        frames: gray-scale video
        flows: magnitude video of optical flows
    """
    # Load video
    cap = cv2.VideoCapture(file_path)
    # Get number of frames
    len_frames = int(cap.get(7))
    # Extract frames from video
    try:
        frames = []
        for i in range(len_frames - 1):
            _, frame = cap.read()
            # frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.reshape(frame, (224, 224, 3))
            frames.append(frame)
    except:
        print("Error: ", file_path, len_frames, i)
    finally:
        frames = np.array(frames)
        cap.release()

    # Get the optical flow of video
    flows = getOpticalFlow(frames)

    result = np.zeros((len(flows), 224, 224, 5))
    result[..., :3] = frames
    result[..., 3:] = flows

    return result


def Save2Npy(file_dir, save_dir):
    """Transfer all the videos and save them into specified directory
    Args:
        file_dir: source folder of target videos
        save_dir: destination folder of output .npy files
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # List the files
    videos = os.listdir(file_dir)
    for v in tqdm(videos):
        # Split video name
        video_name = v.split('.')[0]
        # Get src
        video_path = os.path.join(file_dir, v)
        # Get destination
        save_path = os.path.join(save_dir, video_name + '.npy')
        # Load and preprocess video
        data = Video2Npy(file_path=video_path, resize=(224, 224))
        data = np.uint8(data)
        # Save as .npy file
        np.save(save_path, data)
        # Remove video after converting to npy
        # os.remove(video_path)
    return None


video_dir = '../../Dataset/Images/Unified_videos/Unified'
npy_dir = './Video_Webcam/NPY'

procesar_videos(video_dir, Ctes.FPS)
# Save2Npy(video_dir, npy_dir)
