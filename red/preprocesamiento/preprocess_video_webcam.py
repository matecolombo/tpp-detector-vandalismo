import numpy as np
import cv2


def getOpticalFlowFromCamera(duration):
    """Calculate dense optical flow from the camera feed
    Returns:
        flows_x: the optical flow at x-axis, with the shape of [frames,height,width,channel]
        flows_y: the optical flow at y-axis, with the shape of [frames,height,width,channel]
    """
    # initialize the list of optical flows
    gray_video = []
    # Open the camera
    cap = cv2.VideoCapture(0)

    # Obtener la velocidad de cuadros por segundo
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Calcular la cantidad de cuadros necesarios
    num_frames = int(fps * duration)

    i = 0
    while i < num_frames:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray1_resized = cv2.resize(gray, (224, 224))
        # print(gray1_resized)
        gray_video.append(np.reshape(gray1_resized, (224, 224, 1)))
        # Exit the loop if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        i += 1

    # Release the camera and destroy all windows
    cap.release()
    cv2.destroyAllWindows()

    flows = []
    for i in range(0, len(gray_video) - 1):
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


duration = 5
preprocess_video = getOpticalFlowFromCamera(duration)
print(preprocess_video.shape)
