import pathlib

import cv2


# https://www.youtube.com/watch?v=5cg_yggtkso
def predict_live_cam(model):
    camera = cv2.VideoCapture(0)

    cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"
    clf = cv2.CascadeClassifier(str(cascade_path))

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (255, 0, 255)
    thickness = 2

    print("\nYour main camera is now ON!")
    print("Press 'q' to close!")
    while True:
        _, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = clf.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,  # smaller - more objects
            minSize=(32, 32),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        for (x, y, width, height) in faces:
            face = frame[y:y + height, x:x + width].copy()
            face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face_gray_small = cv2.resize(face_gray, (32, 32), interpolation=cv2.INTER_AREA)

            cnn_face = (face_gray_small / 255) - 0.5
            cnn_face = cnn_face.reshape((-1, 32, 32, 1))

            predict = model.predict(cnn_face, verbose=0)
            # print(f'{predict[0][0]*100:.2f} | {predict[0][1]*100:.2f}')

            text = f'smile: {predict[0][1] * 100:.2f}%'
            cv2.putText(frame, text, (x, y - 5), font, font_scale, color, thickness, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)

        cv2.imshow("predicted face", frame)
        if cv2.waitKey(100) == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()
