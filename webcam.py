import cv2
cap = cv2.VideoCapture(0)  # 0 = webcam default
if not cap.isOpened():
  raise RuntimeError("Kamera tidak dapat dibuka")
while True:
  ok, frame = cap.read()
  if not ok:
    break
  cv2.imshow("Live camera", frame)
  if cv2.waitKey(1) & 0xFF == ord("q"):
    break
  
cap.release()
cv2.destroyAllWindows()