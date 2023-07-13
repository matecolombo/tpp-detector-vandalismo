import numpy as np
import cv2
import os
import datetime
from tqdm import tqdm


class Ctes:
    DURATION_VIDEO_CUTS = 5


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
    print("Cálculo de fps.")
    cap = cv2.VideoCapture(0)
    # Obtener la velocidad de cuadros por segundo
    fps = cap.get(cv2.CAP_PROP_FPS) or fps
    cap.release()
    return fps


def getVideo(video_dir, fps):
    """Calculate dense optical flow from the camera feed
    Args:
        --
    Returns:
        flows_x: the optical flow at x-axis, with the shape of [frames,height,width,channel]
        flows_y: the optical flow at y-axis, with the shape of [frames,height,width,channel]
    """
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)

    resize = (224, 224)

    # Calcular la cantidad de cuadros necesarios
    num_frames = int(fps * Ctes.DURATION_VIDEO_CUTS)

    # Especificar el codec de compresión y la configuración del video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec XVID

    file_name = filename_date()
    save_video_path = os.path.join(video_dir, file_name + '.avi')

    # Open the camera
    cap = cv2.VideoCapture(0)
    print("Se enciende la cámara.")

    # Crear un objeto VideoWriter para guardar el video
    out = cv2.VideoWriter(save_video_path, fourcc, fps, resize)

    i = 0
    while i < num_frames:  # - 1:
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
        out.write(frame)
        # Exit the loop if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        i += 1

    # Release the camera and destroy all windows
    cap.release()
    print("Se apaga la cámara.")
    out.release()
    print("Video almacenado.")
    cv2.destroyAllWindows()
    return save_video_path


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
       # print(frames)
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
        # Get dest
        save_path = os.path.join(save_dir, video_name + '.npy')
        # Load and preprocess video
        data = Video2Npy(file_path=video_path, resize=(224, 224))
        data = np.uint8(data)
        # Save as .npy file
        np.save(save_path, data)
        # Remove video after converting to npy
        # os.remove(video_path)
    return None

'''
fps = get_camera_fps()

video_dir = './Video_Webcam/AVI'
npy_dir = './Video_Webcam/NPY'

video_path = getVideo(video_dir, fps)
Save2Npy(video_dir, npy_dir)
'''