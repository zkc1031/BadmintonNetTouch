import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import filedialog

# MediaPipeの描画ユーティリティと物体検出モデルを準備
mp_drawing = mp.solutions.drawing_utils
mp_object_detection = mp.solutions.object_detection

def select_video_file():
    """
    ファイル選択ダイアログを開いて動画ファイルのパスを取得する関数
    """
    root = tk.Tk()
    root.withdraw()
    filepath = filedialog.askopenfilename(
        title="テストする動画ファイルを選択してください",
        filetypes=[("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*")]
    )
    return filepath

def main():
    """
    メインの処理
    """
    video_path = select_video_file()
    if not video_path:
        print("動画が選択されませんでした。")
        return

    cap = cv2.VideoCapture(video_path)
    
    # 物体検出モデルを初期化
    # model_selection=0は近距離用(～5m)、1は遠距離用(～10m)
    # confidenceを調整すると、検出の感度が変わります (例: 0.3にすると検出しやすくなる)
    with mp_object_detection.ObjectDetector(
        model_selection=0, min_detection_confidence=0.5) as object_detector:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("動画の終端に到達しました。")
                break
            
            # 高速化のため、画像を書き込み不可として参照渡しする
            image.flags.setflags(write=False)
            # MediaPipeはRGBを期待するが、OpenCVはBGRなので色チャンネルを変換
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 物体検出を実行
            results = object_detector.process(image)
            
            # 描画のために、画像を書き込み可能に戻し、色チャンネルを元に戻す
            image.flags.setflags(write=True)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # 検出結果を画像に描画
            if results.detections:
                for detection in results.detections:
                    mp_drawing.draw_detection(image, detection)
            
            # 表示用にウィンドウサイズを調整（任意）
            h, w, _ = image.shape
            display_width = 1280 # ここで見やすいサイズに調整
            if w > display_width:
                aspect_ratio = h / w
                display_height = int(display_width * aspect_ratio)
                image = cv2.resize(image, (display_width, display_height))

            cv2.imshow('MediaPipe Object Detection', image)
            
            # 'q'キーで終了
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()