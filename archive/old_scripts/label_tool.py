import cv2
import os
import tkinter as tk
from tkinter import filedialog
import ntpath

# --- 設定 ---
# 表示するウィンドウの横幅（ピクセル）。ご自身の画面に合わせて調整してください。
# 例：ノートPCなら 960 や 1280 などが見やすいかもしれません。
DISPLAY_WIDTH = 1280

# ラベル付けした画像の保存先ディレクトリ
OUTPUT_DIR = 'data/labeled_frames'
# 分類するクラス（フォルダ名になります）
CLASSES = ['net_touch', 'over_net', 'normal']

# キーボードのどのキーがどのクラスに対応するかを定義
KEY_BINDINGS = {
    't': 'net_touch', # (t)ouch
    'o': 'over_net',  # (o)ver
    'n': 'normal'     # (n)ormal / 何もなし
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
    root.withdraw()
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
        

        # 元のフレームのサイズを取得
        (h, w) = frame.shape[:2]
        # 設定した横幅に合わせて、アスペクト比を維持したまま高さを計算
        r = DISPLAY_WIDTH / float(w)
        dim = (DISPLAY_WIDTH, int(h * r))
        
        # 表示用にフレームをリサイズ
        display_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        
        # 表示用のフレームに情報を書き込む
        info_text = f'File: {video_filename} | Frame: {frame_count}'
        cv2.putText(display_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow('Labeling Tool - Press key to label', display_frame)
        
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('q'):
            print("'q'が押されたため、プログラムを終了します。")
            break
        elif key == ord('s'):
            print(f"Frame {frame_count} をスキップしました。")
            continue
        
        char_key = chr(key)
        if char_key in KEY_BINDINGS:
            class_name = KEY_BINDINGS[char_key]
            save_dir = os.path.join(OUTPUT_DIR, class_name)
            save_filename = f'{video_filename}_frame_{frame_count}.png'
            save_path = os.path.join(save_dir, save_filename)
            
            # 保存するのは、リサイズされていない元の高画質フレーム
            cv2.imwrite(save_path, frame)
            print(f"Saved: Frame {frame_count} -> {class_name} ({save_path})")

    cap.release()
    cv2.destroyAllWindows()
    print("--- ラベル付けを終了しました ---")

if __name__ == '__main__':
    main()