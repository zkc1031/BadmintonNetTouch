import os
import shutil
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf
from tensorflow.keras import layers, models

# --- 設定項目 ---
# 画像サイズ
IMG_WIDTH = 128
IMG_HEIGHT = 128
# バッチサイズとエポック数
BATCH_SIZE = 16
EPOCHS = 10
# クラス名
CLASS_NAMES = ['net_touch', 'over_net', 'other']
NUM_CLASSES = len(CLASS_NAMES)
# データとモデルの保存先
DUMMY_DATA_DIR = 'data/dummy_dataset'
MODEL_SAVE_PATH = 'models/dummy_touch_detector.keras'

def generate_dummy_data():
    """
    ダミーの画像データを生成して、学習用と検証用に分ける関数
    """
    print(f"'{DUMMY_DATA_DIR}'にダミーデータを生成します...")
    if os.path.exists(DUMMY_DATA_DIR):
        shutil.rmtree(DUMMY_DATA_DIR)  # 既存のデータがあれば削除

    # 学習用と検証用のフォルダを作成
    for split in ['train', 'validation']:
        for class_name in CLASS_NAMES:
            os.makedirs(os.path.join(DUMMY_DATA_DIR, split, class_name), exist_ok=True)

    # 各クラスのダミー画像を生成
    for class_idx, class_name in enumerate(CLASS_NAMES):
        for i in range(70): # 1クラスあたり70枚生成
            img = Image.new('RGB', (IMG_WIDTH, IMG_HEIGHT), color = 'white')
            draw = ImageDraw.Draw(img)

            # ノイズを追加
            for _ in range(100):
                x, y = np.random.randint(0, IMG_WIDTH), np.random.randint(0, IMG_HEIGHT)
                draw.point((x, y), fill='gray')

            if class_name == 'net_touch':
                # ネットタッチ：中央に太い水平線
                y_center = IMG_HEIGHT // 2
                draw.line((0, y_center, IMG_WIDTH, y_center), fill='black', width=5)
                draw.ellipse((IMG_WIDTH//2-10, y_center-10, IMG_WIDTH//2+10, y_center+10), fill='red')

            elif class_name == 'over_net':
                # オーバーネット：中央の線を越える図形
                y_center = IMG_HEIGHT // 2
                draw.line((0, y_center, IMG_WIDTH, y_center), fill='gray', width=1)
                draw.rectangle((IMG_WIDTH//2-20, y_center-40, IMG_WIDTH//2+20, y_center-10), outline='blue', width=3)

            else: # other
                # その他：特に特徴のない画像
                pass

            # 50枚を学習用、20枚を検証用に保存
            split = 'train' if i < 50 else 'validation'
            img_path = os.path.join(DUMMY_DATA_DIR, split, class_name, f'{i}.png')
            img.save(img_path)

    print("ダミーデータの生成が完了しました。")

def build_model():
    """
    シンプルなCNNモデルを構築する関数
    """
    model = models.Sequential([
        # 入力層: 画像のサイズとチャンネル数（RGBなので3）を指定
        layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        # データ拡張（学習データのかさ増しと汎化性能向上）
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        # Convolution層とPooling層
        layers.Rescaling(1./255), # ピクセル値を0-1の範囲に正規化
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        # 全結合層
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        # 出力層: クラス数に応じたユニット数とsoftmax活性化関数
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    # モデルのコンパイル
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', # ラベルが整数の場合に利用
                  metrics=['accuracy'])
    return model

def main():
    """
    メインの実行関数
    """
    # 1. ダミーデータの生成
    generate_dummy_data()

    # 2. データの読み込み
    print("\nデータセットを読み込んでいます...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DUMMY_DATA_DIR, 'train'),
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        label_mode='int' # ラベルを整数（0, 1, 2）として読み込む
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DUMMY_DATA_DIR, 'validation'),
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        label_mode='int'
    )
    # パフォーマンス向上のための設定
    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    print("データセットの準備が完了しました。")

    # 3. モデルの構築
    model = build_model()
    print("\nモデルの構造:")
    model.summary()

    # 4. モデルの学習
    print("\nモデルの学習を開始します...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )
    print("モデルの学習が完了しました。")

    # 5. モデルの保存
    os.makedirs('models', exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    print(f"\n学習済みモデルを '{MODEL_SAVE_PATH}' に保存しました。")


if __name__ == '__main__':
    main()