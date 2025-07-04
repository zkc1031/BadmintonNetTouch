import os
import random
import shutil

# --- 設定 ---
# ラベル付けした画像が入っている元のフォルダ
SOURCE_DIR = 'data/labeled_frames_final' # or 'data/labeled_frames_advanced'など、実際に使っているフォルダ名

# 分割後のデータを保存する先のフォルダ
OUTPUT_DIR = 'data/split_dataset'

# 分割の比率 (学習用, 検証用, テスト用) - 合計が1.0になるように
SPLIT_RATIO = (0.8, 0.1, 0.1)

# 分類するクラス名（手動で設定するか、SOURCE_DIRから自動で取得するか選べる）
# CLASSES = ['net_touch', 'over_net', 'normal']

def split_data():
    """
    データを学習・検証・テスト用に分割する関数
    """
    print(f"'{SOURCE_DIR}' からデータを読み込んでいます...")
    if not os.path.exists(SOURCE_DIR):
        print(f"エラー: 元となるデータフォルダ '{SOURCE_DIR}' が見つかりません。")
        return

    # 出力フォルダが既にあれば、一度削除して作り直す（重複を防ぐため）
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
        print(f"既存の出力フォルダ '{OUTPUT_DIR}' を削除しました。")

    # クラス名のリストを自動で取得
    try:
        classes = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]
        if not classes:
            print(f"エラー: '{SOURCE_DIR}' の中にクラスごとのサブフォルダが見つかりません。")
            return
        print(f"検出されたクラス: {classes}")
    except Exception as e:
        print(f"エラー: クラスフォルダの読み込み中に問題が発生しました: {e}")
        return

    # train, validation, test のフォルダを作成
    train_dir = os.path.join(OUTPUT_DIR, 'train')
    validation_dir = os.path.join(OUTPUT_DIR, 'validation')
    test_dir = os.path.join(OUTPUT_DIR, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(validation_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for class_name in classes:
        # 各クラスの出力用サブフォルダを作成
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(validation_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

        # ファイルリストを取得
        class_source_dir = os.path.join(SOURCE_DIR, class_name)
        all_files = [f for f in os.listdir(class_source_dir) if os.path.isfile(os.path.join(class_source_dir, f))]
        
        # ファイルリストをランダムにシャッフル
        random.shuffle(all_files)
        
        # ファイルを分割
        num_files = len(all_files)
        num_train = int(num_files * SPLIT_RATIO[0])
        num_validation = int(num_files * SPLIT_RATIO[1])
        
        train_files = all_files[:num_train]
        validation_files = all_files[num_train : num_train + num_validation]
        test_files = all_files[num_train + num_validation:]

        # ファイルをコピー
        for file in train_files:
            shutil.copy(os.path.join(class_source_dir, file), os.path.join(train_dir, class_name, file))
        for file in validation_files:
            shutil.copy(os.path.join(class_source_dir, file), os.path.join(validation_dir, class_name, file))
        for file in test_files:
            shutil.copy(os.path.join(class_source_dir, file), os.path.join(test_dir, class_name, file))

        print(f"クラス '{class_name}': "
              f"学習用 {len(train_files)}枚, "
              f"検証用 {len(validation_files)}枚, "
              f"テスト用 {len(test_files)}枚 に分割しました。")

    print(f"\nデータ分割が完了しました。出力先: '{OUTPUT_DIR}'")

if __name__ == '__main__':
    split_data()