# ダミーモデル用コード
# models/dummy_model.py
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ダミーデータ作成
X, y = make_classification(n_samples=100, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# モデル訓練
model = LogisticRegression()
model.fit(X_train, y_train)

# 評価
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
