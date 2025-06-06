import cv2
import os

video_path = 'sample.mp4'  # 動画ファイルのパス
label_output = 'labels.txt'  # ラベル保存先ファイル

cap = cv2.VideoCapture(video_path)
frame_number = 0

# ラベル保存用ファイルを開く
with open(label_output, 'w') as f:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('Frame', frame)
        key = cv2.waitKey(0)  # キー入力を待つ

        if key == ord('n'):
            f.write(f'{frame_number},normal\n')
        elif key == ord('o'):
            f.write(f'{frame_number},overnet\n')
        elif key == ord('t'):
            f.write(f'{frame_number},nettouch\n')
        elif key == ord('q'):
            break

        frame_number += 1

cap.release()
cv2.destroyAllWindows()
