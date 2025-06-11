import cv2
import mediapipe as mp
import os
import tkinter as tk
from tkinter import filedialog
import ntpath
from scipy.spatial import distance as dist
import numpy as np

# MediaPipeの新しいAPIのためのインポート
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- MediaPipeの準備 ---
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# --- 設定 ---
MODEL_PATH = 'efficientdet_lite0.tflite'
OUTPUT_DIR = 'data/labeled_frames_advanced'
CLASSES = ['net_touch', 'over_net', 'normal']
KEY_BINDINGS = {'t': 'net_touch', 'o': 'over_net', 'n': 'normal'}
DISTANCE_THRESHOLD = 150
DISPLAY_WIDTH = 1280

# --- 補助関数 ---

def setup_directories():
    """保存用のディレクトリがなければ作成する"""
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    for class_name in CLASSES:
        if not os.path.exists(os.path.join(OUTPUT_DIR, class_name)):
            os.makedirs(os.path.join(OUTPUT_DIR, class_name))
    print(f"保存用ディレクトリ '{OUTPUT_DIR}' の準備が完了しました。")

def select_video_file():
    """ファイル選択ダイアログを開いて動画ファイルのパスを取得する"""
    root = tk.Tk()
    root.withdraw()
    filepath = filedialog.askopenfilename(title="ラベル付けする動画ファイルを選択", filetypes=[("Video files", "*.mp4 *.avi *.mov")])
    return filepath

def get_bbox_center(bbox):
    """MediaPipeのバウンディングボックスの中心座標を返す"""
    return (bbox.origin_x + bbox.width // 2, bbox.origin_y + bbox.height // 2)

def find_net_tape_line(roi_image):
    """ROI画像からハフ変換で最も長い水平に近い直線を検出する"""
    if roi_image is None or roi_image.size == 0:
        return None
    gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=40, maxLineGap=10)
    
    if lines is not None:
        longest_line = None
        max_len = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # 水平に近い線に限定
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if angle < 10 or angle > 170:
                length = dist.euclidean((x1, y1), (x2, y2))
                if length > max_len:
                    max_len = length
                    longest_line = (x1, y1, x2, y2)
        return longest_line
    return None

def find_racket_contour(frame, bbox):
    """ラケットのBBoxから輪郭を検出する"""
    x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
    racket_roi = frame[y:y+h, x:x+w]

    if racket_roi.size == 0:
        return None
    
    gray = cv2.cvtColor(racket_roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        largest_contour[:, :, 0] += x
        largest_contour[:, :, 1] += y
        return largest_contour
    return None


# --- メイン処理 ---
def main():
    if not os.path.exists(MODEL_PATH):
        print(f"エラー: モデルファイル '{MODEL_PATH}' が見つかりません。")
        print("プロジェクトフォルダの直下にダウンロードしたモデルを置いてください。")
        return

    setup_directories()
    video_path = select_video_file()
    if not video_path:
        print("動画が選択されませんでした。")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"エラー: {video_path} を開けませんでした。")
        return

    ret, first_frame = cap.read()
    if not ret:
        print("動画からフレームを読み込めませんでした。")
        return
    
    (h, w) = first_frame.shape[:2]
    r = DISPLAY_WIDTH / float(w)
    dim = (DISPLAY_WIDTH, int(h * r))
    first_frame_resized = cv2.resize(first_frame, dim)

    cv2.putText(first_frame_resized, "Mouse drag to select Net ROI, then press ENTER or SPACE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    net_roi_resized = cv2.selectROI("Select Net ROI", first_frame_resized, fromCenter=False, showCrosshair=True)
    
    # ROIが選択されなかった場合（ESCキーなど）
    if net_roi_resized == (0, 0, 0, 0):
        print("ROIが選択されませんでした。プログラムを終了します。")
        cap.release()
        cv2.destroyAllWindows()
        return

    rx, ry, rw, rh = [int(c / r) for c in net_roi_resized]
    net_roi_center = (rx + rw // 2, ry + rh // 2)
    cv2.destroyWindow("Select Net ROI")

    video_filename = ntpath.basename(video_path).split('.')[0]
    frame_count = 0

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    object_options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.5, category_allowlist=["sports racket"])
    object_detector = vision.ObjectDetector.create_from_options(object_options)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame_count += 1
            display_frame = frame.copy()
            
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pose_results = pose.process(mp_image.numpy_view())
            detection_result = object_detector.detect(mp_image)
            
            pause_for_labeling = False
            racket_contour, net_line = None, None

            if pose_results.pose_landmarks and detection_result.detections:
                racket_detection = detection_result.detections[0]
                racket_bbox = racket_detection.bounding_box
                racket_center = get_bbox_center(racket_bbox)
                
                landmarks = pose_results.pose_landmarks.landmark
                left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
                right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
                
                left_wrist_pos = (int(left_wrist.x * w), int(left_wrist.y * h))
                right_wrist_pos = (int(right_wrist.x * w), int(right_wrist.y * h))
                
                dist_to_left = dist.euclidean(racket_center, left_wrist_pos)
                dist_to_right = dist.euclidean(racket_center, right_wrist_pos)
                
                racket_hand_pos = left_wrist_pos if dist_to_left < dist_to_right else right_wrist_pos
                
                if dist.euclidean(racket_hand_pos, net_roi_center) < DISTANCE_THRESHOLD + (rw+rh)/2:
                    pause_for_labeling = True
                    net_roi_img = frame[ry:ry+rh, rx:rx+rw]
                    line_coords = find_net_tape_line(net_roi_img)
                    if line_coords:
                        x1, y1, x2, y2 = line_coords
                        net_line = ((x1 + rx, y1 + ry), (x2 + rx, y2 + ry))
                    
                    racket_contour = find_racket_contour(frame, racket_bbox)

            if pause_for_labeling:
                if net_line: cv2.line(display_frame, net_line[0], net_line[1], (0, 255, 255), 3)
                if racket_contour is not None: cv2.drawContours(display_frame, [racket_contour], -1, (0, 255, 0), 2)
            
            display_frame_resized = cv2.resize(display_frame, dim)
            cv2.imshow('Advanced Labeling Tool', display_frame_resized)
            
            wait_key_time = 0 if pause_for_labeling else 1
            key = cv2.waitKey(wait_key_time) & 0xFF

            if key == ord('q'): break
            if pause_for_labeling:
                if key == ord('s'): continue
                char_key = chr(key)
                if char_key in KEY_BINDINGS:
                    class_name = KEY_BINDINGS.get(char_key)
                    if class_name:
                        highlighted_frame = frame.copy()
                        if net_line: cv2.line(highlighted_frame, net_line[0], net_line[1], (0, 255, 255), 3)
                        if racket_contour is not None: cv2.drawContours(highlighted_frame, [racket_contour], -1, (0, 255, 0), 2)

                        save_dir = os.path.join(OUTPUT_DIR, class_name)
                        save_filename = f'{video_filename}_frame_{frame_count}_advanced.png'
                        cv2.imwrite(os.path.join(save_dir, save_filename), highlighted_frame)
                        print(f"Saved: Frame {frame_count} -> {class_name}")

    cap.release()
    cv2.destroyAllWindows()
    print("--- ツールを終了しました ---")

if __name__ == '__main__':
    main()