import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os

# --- 設定 ---
# データセットのパス（データを分割した後のパス）
BASE_DATA_PATH = 'data/split_dataset'
TRAIN_DIR = os.path.join(BASE_DATA_PATH, 'train')
VALIDATION_DIR = os.path.join(BASE_DATA_PATH, 'validation')

# モデルのパラメータ
IMG_WIDTH = 224 # MobileNetV2は224x224を基本とする
IMG_HEIGHT = 224
BATCH_SIZE = 32
EPOCHS = 50 # EarlyStoppingがあるので多めに設定してもOK
NUM_CLASSES = 3 # net_touch, over_net, normal

# 保存するモデルのパス
MODEL_SAVE_PATH = 'models/badminton_net_detector_v1.keras'


def build_model():
    """転移学習を用いたモデルを構築する"""
    # 1. 事前学習済みモデル（MobileNetV2）を土台としてロード
    # include_top=False は、最後の全結合層（1000クラス分類用）を含めないという意味
    base_model = MobileNetV2(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
                             include_top=False,
                             weights='imagenet')

    # 土台のモデルの重みは、最初は凍結して再学習させない
    base_model.trainable = False

    # 2. データ拡張（Data Augmentation）層の作成
    data_augmentation = models.Sequential([
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.2),
    ], name='data_augmentation')

    # 3. 新しいモデルの構築
    inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = data_augmentation(inputs) # まずデータ拡張
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x) # MobileNetV2用の前処理
    x = base_model(x, training=False) # 土台モデル（重みは凍結）
    x = layers.GlobalAveragePooling2D()(x) # 特徴量を平坦化
    x = layers.Dropout(0.2)(x) # 過学習を防ぐ
    # 最後の出力層（3クラス分類）
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)

    # 4. モデルのコンパイル
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def main():
    """メインの学習処理"""
    # データセットの準備
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        shuffle=True,
        batch_size=BATCH_SIZE,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        label_mode='categorical'
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        VALIDATION_DIR,
        shuffle=True,
        batch_size=BATCH_SIZE,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        label_mode='categorical'
    )
    
    # パフォーマンス向上のための設定
    train_ds = train_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = validation_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    # モデルの構築
    model = build_model()
    model.summary()

    # コールバックの準備
    # 最も性能が良いモデルを保存
    checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, 
                                 monitor='val_accuracy', 
                                 save_best_only=True, 
                                 mode='max',
                                 verbose=1)
    # 性能が改善しなくなったら早期終了
    early_stopping = EarlyStopping(monitor='val_accuracy', 
                                   patience=10, # 10エポック改善がなければ停止
                                   restore_best_weights=True,
                                   verbose=1)

    print("\n--- 本番用の学習を開始します ---")
    
    # 学習の実行
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=[checkpoint, early_stopping]
    )
    
    print(f"\n学習が完了しました。最も性能の良いモデルが '{MODEL_SAVE_PATH}' に保存されています。")


if __name__ == '__main__':
    # データフォルダが存在するか簡単なチェック
    if not os.path.exists(TRAIN_DIR) or not os.path.exists(VALIDATION_DIR):
        print("エラー: 学習データまたは検証データフォルダが見つかりません。")
        print(f"'{TRAIN_DIR}' と '{VALIDATION_DIR}' を確認してください。")
        print("まず、データを集めて、自動分割スクリプトを実行してください。")
    else:
        main()