print("--- デバッグ開始 ---")

try:
    print("1. osをインポート中...")
    import os
    print("   os OK")

    print("2. tkinterをインポート中...")
    import tkinter as tk
    from tkinter import filedialog
    print("   tkinter OK")

    print("3. ntpathをインポート中...")
    import ntpath
    print("   ntpath OK")
    
    print("4. scipyをインポート中...")
    from scipy.spatial import distance as dist
    print("   scipy OK")

    print("5. cv2 (OpenCV)をインポート中...")
    import cv2
    print("   cv2 (OpenCV) OK")

    print("6. mediapipeをインポート中...")
    import mediapipe as mp
    print("   mediapipe OK")
    
    print("\n--- すべてのライブラリのインポートに成功しました ---")

except Exception as e:
    print(f"\n--- エラー発生 ---")
    print(f"エラー内容: {e}")