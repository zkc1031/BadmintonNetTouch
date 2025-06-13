import cv2
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO

# --- 設定 ---
DISPLAY_WIDTH = 1280

def select_video_file():
    """ファイル選択ダイアログを開く"""
    root = tk.Tk()
    root.withdraw()
    filepath = filedialog.askopenfilename(title="テストする動画ファイルを選択", filetypes=[("Video files", "*.mp4 *.avi *.mov")])
    return filepath

def main():
    video_path = select_video_file()
    if not video_path:
        print("動画が選択されませんでした。")
        return

    # YOLOv8の骨格推定モデルをロード
    # 'yolov8n-pose.pt' -> nはnanoの略。最速・最軽量モデル。
    # より高精度なモデルを使いたい場合は 'yolov8s-pose.pt' (small) などに変更。
    model = YOLO('yolov8n-pose.pt')

    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # モデルで骨格推定を実行
        results = model(frame, stream=True)

        # 結果をフレームに描画
        for r in results:
            # .plot()メソッド一発で、バウンディングボックスと骨格を描画してくれる優れもの
            annotated_frame = r.plot()
            
            # 表示用にリサイズ
            (h, w, _) = annotated_frame.shape
            if w > DISPLAY_WIDTH:
                r = DISPLAY_WIDTH / float(w)
                dim = (DISPLAY_WIDTH, int(h * r))
                annotated_frame = cv2.resize(annotated_frame, dim)

            cv2.imshow("YOLOv8 Pose Estimation", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()