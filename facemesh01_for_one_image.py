from cv2 import cv2
import mediapipe as mp

# Read images with OpenCV.

image = cv2.imread(r'E:\TRPcodes\mediapipe\trp_02.jpg')
cv2.imshow('image', image)
cv2.waitKey(0)  
# Preview the images.
mp_face_mesh = mp.solutions.face_mesh
# Prepare DrawingSpec for drawing the face landmarks later.
mp_drawing = mp.solutions.drawing_utils 
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

with mp_face_mesh.FaceMesh(
  static_image_mode=True,
  max_num_faces=2,
  min_detection_confidence=0.5) as face_mesh:
  #for name, image in images.items():
    # Convert the BGR image to RGB and process it with MediaPipe Face Mesh.
  results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Draw face landmarks of each face.
  print(f'Face landmarks of {image}:')
    #if not results.multi_face_landmarks:
     # continue
  annotated_image = image.copy()
    #for face_landmarks in results.multi_face_landmarks:
  mp_drawing.draw_landmarks(
          image=annotated_image,
          landmark_list=results, #face_landmarks,
          connections=mp_face_mesh.FACE_CONNECTIONS,
          landmark_drawing_spec=drawing_spec,
          connection_drawing_spec=drawing_spec)
  cv2_imshow(annotated_image)