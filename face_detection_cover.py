import cv2
import sys
import os


if  not (os.path.isfile('goturn.caffemodel') and os.path.isfile('goturn.prototxt')):
    errorMsg = '''
    Could not find GOTURN model in current directory.
    Please ensure goturn.caffemodel and goturn.prototxt are in the current directory
    '''

    print(errorMsg)
    sys.exit()
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
face_cascade = cv2.CascadeClassifier(r'C:\Users\HP001\OneDrive - De Haagse Hogeschool\Documenten\ObjDet\project1\venv\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
# Read the frame
_, img = cap.read()
    # Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)
cap.release()

tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
tracker_type = tracker_types[2]

if int(minor_ver) < 3:
    tracker = cv2.Tracker_create(tracker_type)
else:
    if tracker_type == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    elif tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    elif tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    elif tracker_type == 'TLD':
        tracker = cv2.legacy.TrackerTLD_create()
    elif tracker_type == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    elif tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()
    elif tracker_type == 'MOSSE':
        tracker = cv2.legacy.TrackerMOSSE_create()
    elif tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()

video = cv2.VideoCapture(0)
#video = cv2.VideoCapture(0) # for using CAM

# Exit if video not opened.
if not video.isOpened():
  print("Could not open video")
  sys.exit()
print("h")
# Read first frame.
ok, frame = video.read()
if not ok:
  print ('Cannot read video file')
  sys.exit()
bbox=(100,100,100,100)
  # Define an initial bounding box
for (x,y,w,h) in faces:
    bbox = (x, y, w, h)

  # Uncomment the line below to select a different bounding box
 # bbox = cv2.selectROI(frame, False)
print(bbox)
  # Initialize tracker with first frame and bounding box
ok = tracker.init(frame, bbox)
print("hallo")
teller=0
face_detect=False
while True:
  # Read a new frame
  ok, frame = video.read()
  if not ok:
      break

  # Start timer
  timer = cv2.getTickCount()

  # Update tracker
  ok, bbox = tracker.update(frame)

  # Calculate Frames per second (FPS)
  fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

  # Draw bounding box
  if ok:
      # Tracking success
      p1 = (int(bbox[0]), int(bbox[1]))
      p2 = (int(bbox[0] + bbox[2]+55), int(bbox[1] + bbox[3]+55))
      cv2.rectangle(frame, p1, p2, (255, 0, 0), -1, 1)

  else:
      frame[:][:][:]=(0,0,255)
      #tracking failure
      p1 = (int(bbox[0]), int(bbox[1]))
      p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
      cv2.rectangle(frame, p1, p2, (255, 0, 0), -1, 1)
      #frame[:][:][:]= (0,0,255)
      teller+=1
      if(teller>0):
        teller=0
        _, img = video.read()
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect the faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                bbox = (x, y, w, h)
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), -1, 1)
            face_detect = True

            tracker2= cv2.TrackerKCF_create()
            tracker = tracker2
            ok = tracker.init(frame, bbox)
        else:
            face_detect = False

        if(face_detect):
            ok=True

  # Display tracker type on frame
  cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

  # Display FPS on frame
  cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);
  # Display result
  cv2.imshow("Tracking", frame)

  # Exit if ESC pressed
  if cv2.waitKey(1) & 0xFF == ord('q'):  # if press SPACE bar
      break

video.release()
cv2.destroyAllWindows()