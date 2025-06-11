print("--- 詳細デバッグ開始 ---")
print("MediaPipeのどの機能が問題かを特定します。")

try:
    print("\n1. mediapipeをインポート中...")
    import mediapipe as mp
    print("   -> mediapipe のインポートOK")

    print("\n2. drawing_utils を準備中...")
    mp_drawing = mp.solutions.drawing_utils
    print("   -> drawing_utils の準備OK")

    print("\n3. pose (骨格推定) を準備中...")
    mp_pose = mp.solutions.pose
    print("   -> pose (骨格推定) の準備OK")

    print("\n4. object_detection (物体検出) を準備中...")
    mp_object_detection = mp.solutions.object_detection
    print("   -> object_detection (物体検出) の準備OK")
    
    print("\n--- すべての機能の準備に成功しました ---")

except Exception as e:
    print(f"\n--- エラー発生 ---")
    print(f"エラー内容: {e}")