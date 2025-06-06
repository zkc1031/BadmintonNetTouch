# ラベル付けツール
import cv2
import os
import tkinter as tk
from tkinter import filedialog
import ntpath

# --- 設定 ---
# ラベル付けした画像の保存先ディレクトリ
OUTPUT_DIR = 'data/labeled_frames'
# 分類するクラス（フォルダ名になります）
CLASSES = ['net_touch', 'over_net', 'normal']

# キーボードのどのキーがどのクラスに対応するかを定義
KEY_BINDINGS = {
    't': 'net_touch', # (t)ouch
    'o': 'over_net',  # (o)ver
    'n': 'normal'      # (n)ormal / 何もなし
}
# --- 設定ここまで ---

def setup_directories():
    """
    保存用のディレクトリがなければ作成する関数
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    for class_name in CLASSES:
        class_dir = os.path.join(OUTPUT_DIR, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
    print("保存用ディレクトリの準備が完了しました。")

def select_video_file():
    """
    ファイル選択ダイアログを開いて動画ファイルのパスを取得する関数
    """
    root = tk.Tk()
    root.withdraw()  # 小さなウィンドウが表示されるのを防ぐ
    filepath = filedialog.askopenfilename(
        title="ラベル付けする動画ファイルを選択してください",
        filetypes=[("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*")]
    )
    return filepath

def main():
    """
    メインの処理を行う関数
    """
    setup_directories()
    video_path = select_video_file()

    if not video_path:
        print("動画ファイルが選択されませんでした。プログラムを終了します。")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"エラー: {video_path} を開けませんでした。")
        return
    
    video_filename = ntpath.basename(video_path).split('.')[0]
    frame_count = 0
    
    print("\n--- ラベル付けを開始します ---")
    print("t: ネットタッチ | o: オーバーネット | n: 何もなし | s: スキップ | q: 終了")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("動画の最後まで到達しました。")
            break
        
        frame_count += 1
        
        # 表示用のフレームをコピーして、情報を書き込む
        display_frame = frame.copy()
        info_text = f'File: {video_filename} | Frame: {frame_count}'
        cv2.putText(display_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow('Labeling Tool - Press key to label', display_frame)
        
        # キー入力を待つ (0は無限待ち)
        key = cv2.waitKey(0) & 0xFF
        
        # 'q'が押されたら終了
        if key == ord('q'):
            print("'q'が押されたため、プログラムを終了します。")
            break
        # 's'が押されたらスキップ（次のフレームへ）
        elif key == ord('s'):
            print(f"Frame {frame_count} をスキップしました。")
            continue
        
        # キーが登録されているか確認
        char_key = chr(key)
        if char_key in KEY_BINDINGS:
            class_name = KEY_BINDINGS[char_key]
            save_dir = os.path.join(OUTPUT_DIR, class_name)
            
            # ファイル名を決定 (例: my_video_frame_123.png)
            save_filename = f'{video_filename}_frame_{frame_count}.png'
            save_path = os.path.join(save_dir, save_filename)
            
            # 元のフレームを保存 (文字が書き込まれていない方)
            cv2.imwrite(save_path, frame)
            print(f"Saved: Frame {frame_count} -> {class_name} ({save_path})")

    cap.release()
    cv2.destroyAllWindows()
    print("--- ラベル付けを終了しました ---")

if __name__ == '__main__':
    main()