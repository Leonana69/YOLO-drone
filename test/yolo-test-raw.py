from ultralytics import YOLOWorld, YOLO
import cv2

# model = YOLOWorld('yolov8s-worldv2.pt')
model = YOLO('yolov8s.pt')
# model.set_class(['suitcase', 'person'])

def format_result(yolo_result):
    if yolo_result.probs is not None:
        print('Warning: Classify task do not support `tojson` yet.')
        return
    formatted_result = []
    data = yolo_result.boxes.data.cpu().tolist()
    h, w = yolo_result.orig_shape
    for i, row in enumerate(data):  # xyxy, track_id if tracking, conf, class_id
        box = {'x1': round(row[0] / w, 2), 'y1': round(row[1] / h, 2), 'x2': round(row[2] / w, 2), 'y2': round(row[3] / h, 2)}
        conf = row[-2]
        class_id = int(row[-1])

        name = yolo_result.names[class_id]
        if yolo_result.boxes.is_track:
            # result['track_id'] = int(row[-3])  # track ID
            name = f'{name}_{int(row[-3])}'
        result = {'name': name, 'confidence': round(conf, 2), 'box': box}
        
        if yolo_result.masks:
            x, y = yolo_result.masks.xy[i][:, 0], yolo_result.masks.xy[i][:, 1]  # numpy array
            result['segments'] = {'x': (x / w).tolist(), 'y': (y / h).tolist()}
        if yolo_result.keypoints is not None:
            x, y, visible = yolo_result.keypoints[i].data[0].cpu().unbind(dim=1)  # torch Tensor
            result['keypoints'] = {'x': (x / w).tolist(), 'y': (y / h).tolist(), 'visible': visible.tolist()}
        formatted_result.append(result)
    return formatted_result

def plot_results(frame, results):
    if results is None:
        return
    def str_float_to_int(value, multiplier):
        return int(float(value) * multiplier)
    w, h = frame.shape[1], frame.shape[0]
    for result in results:
        box = result["box"]
        #draw.rectangle((str_float_to_int(box["x1"], w), str_float_to_int(box["y1"], h), str_float_to_int(box["x2"], w), str_float_to_int(box["y2"], h)),
                    #fill=None, outline='blue', width=4)
        cv2.rectangle(frame, (str_float_to_int(box["x1"], w), str_float_to_int(box["y1"], h)), (str_float_to_int(box["x2"], w), str_float_to_int(box["y2"], h)), (255, 0, 0), 2)
        #draw.text((str_float_to_int(box["x1"], w), str_float_to_int(box["y1"], h) - 50), result["name"], fill='red', font=font)
        cv2.putText(frame, result["name"], (str_float_to_int(box["x1"], w), str_float_to_int(box["y1"], h) - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f'{result["confidence"]:.2f}', (str_float_to_int(box["x1"], w), str_float_to_int(box["y1"], h) - 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
# open camera
cap = cv2.VideoCapture(0)
from filterpy.kalman import KalmanFilter
import numpy as np
def init_filter():
    kf = KalmanFilter(dim_x=4, dim_z=2)  # 4 state dimensions (x, y, vx, vy), 2 measurement dimensions (x, y)
    kf.F = np.array([[1, 0, 1, 0],  # State transition matrix
                    [0, 1, 0, 1],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
    kf.H = np.array([[1, 0, 0, 0],  # Measurement function
                    [0, 1, 0, 0]])
    kf.R *= 1  # Measurement uncertainty
    kf.P *= 1000  # Initial uncertainty
    kf.Q *= 0.01  # Process uncertainty
    return kf

kf = init_filter()
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # detect
    inference = model.track(frame, conf=0.1)
    print(inference)
    result = format_result(inference[0])
    # result = format_result(model.track(frame, conf=0.1, persist=True, tracker="bytetrack.yaml")[0])
    plot_results(frame, result)
    has_person = False
    for item in result:
        if item["name"].startswith('suitcase'):
            has_person = True
            loc_x = (item["box"]["x1"] + item["box"]["x2"]) / 2
            loc_y = (item["box"]["y1"] + item["box"]["y2"]) / 2
            kf.update((loc_x, loc_y))

    kf.predict()

    print(kf.x, frame.shape[0], frame.shape[1])
    cv2.circle(frame, (int(kf.x[0] * frame.shape[1]), int(kf.x[1] * frame.shape[0])), 5, (0, 255, 0), -1)
    # print(model(frame, conf=0.01))
    # exit(0)
    # display
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break