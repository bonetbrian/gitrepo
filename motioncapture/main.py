import cv2, pandas
from datetime import datetime 

# Initializing variables
static_back = None
motion_list = [None, None] 
time_list = [] 
df = pandas.DataFrame(columns=["Start", "End"]) 

# Capturing video
video = cv2.VideoCapture(0) 

while True: 
    check, frame = video.read() 
    motion = 0

    # Converting color image to gray_scale image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    gray = cv2.GaussianBlur(gray, (21, 21), 0) 

    if static_back is None: 
        static_back = gray 
        continue

    # Calculate difference between static background and current frame
    diff_frame = cv2.absdiff(static_back, gray) 
    thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)[1] 
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2) 

    # Finding contour of moving object
    contours, _ = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 

    for contour in contours: 
        if cv2.contourArea(contour) < 10000: 
            continue
        motion = 1

        # Drawing rectangle around the moving object
        (x, y, w, h) = cv2.boundingRect(contour) 
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3) 

    # Update motion list and time_list based on motion
    motion_list.append(motion) 
    if motion_list[-1] == 1 and motion_list[-2] == 0: 
        time_list.append(datetime.now()) 
    if motion_list[-1] == 0 and motion_list[-2] == 1: 
        time_list.append(datetime.now()) 

    # Displaying the current frame with detected motion
    cv2.imshow("Motion Detector", frame) 

    # Check if the window is closed
    if cv2.getWindowProperty("Motion Detector", cv2.WND_PROP_VISIBLE) < 1:
        break

    # Stop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        if motion == 1: 
            time_list.append(datetime.now()) 
        break

# Create DataFrame from time_list
for i in range(0, len(time_list), 2): 
    df = df.append({"Start": time_list[i], "End": time_list[i + 1]}, ignore_index=True) 

# Save DataFrame to CSV
df.to_csv("Time_of_movements.csv") 

video.release() 
cv2.destroyAllWindows() 
