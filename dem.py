from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2
import time 


model = YOLO("pen.pt")
cap = cv2.VideoCapture(0)

pTime = 0
cTime = 0

assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Define line points
line_points = [(1, 300), (1080, 300)]

# Video writer
video_writer = cv2.VideoWriter("object_counting_output.avi",
                       cv2.VideoWriter_fourcc(*'mp4v'),
                       fps,
                       (w, h))

# Init Object Counter
counter = object_counter.ObjectCounter()
counter.set_args(view_img=True,
                 reg_pts=line_points,
                 classes_names=model.names,
                 draw_tracks=True,
                 line_thickness=2)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    
    tracks = model.track(im0, persist=True, show=False)
    cv2.putText(im0, f'FPS: {fps:.2f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    im0 = counter.start_counting(im0, tracks)
    video_writer.write(im0)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()