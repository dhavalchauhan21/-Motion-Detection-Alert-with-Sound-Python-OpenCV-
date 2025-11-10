import cv2
from playsound import playsound
import time
import datetime


face_cascade = cv2.CascadeClassifier(r'C:/Users/Dhaval/Documents/Face detection/haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)

motion_detected = False
log = []
last_sound_time = 0  # time when the last sound was played
sound_cooldown = 10   # seconds to wait before playing sound again
while True:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

    # Motion detection
    blur = cv2.GaussianBlur(gray, (21, 21), 0)
    if 'prev' in locals():
        diff = cv2.absdiff(prev, blur)
        thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
        motion_area = thresh.sum() // 255

        if motion_area > 1000 and not motion_detected:

            motion_detected = True
            t = datetime.datetime.now().strftime('%H:%M:%S')
            log.append(f"Motion started at {t}")
        elif motion_area <= 5000 and motion_detected:
            motion_detected = False
            t = datetime.datetime.now().strftime('%H:%M:%S')
            log.append(f"Motion stopped at {t}")
            print("Motion detected at", t)
            playsound("C:/Users/Dhaval/Documents/Face detection/vine-boom-sound-effect(chosic.com).wav")




        cv2.imshow("Motion", thresh)

    prev = blur
    cv2.imshow("Live Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

# Save motion log
with open("motion_log.txt", "w") as f:
    for entry in log:
        f.write(entry + "\n")
