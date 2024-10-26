import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import joblib
from imblearn.over_sampling import SMOTE

# 1. 导入数据
file_path = 'similarity_results.csv'  # 使用你的 CSV 文件路径
df = pd.read_csv(file_path)

# 2. 清理相似度特征列，去掉方括号
df_cleaned = df.copy()
for col in df_cleaned.columns[2:]:
    df_cleaned[col] = df_cleaned[col].apply(lambda x: float(x.strip('[]')))

# 3. 准备数据，X 是相似度特征，y 是真实类别
X = df_cleaned.drop(columns=['image_name', 'true_class'])
y = df_cleaned['true_class']

# 4. 将字符串标签转换为整数
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 5. 将数据划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 6. 归一化特征
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. 使用SMOTE对训练集进行过采样
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)

# 8. 定义更多模型
clf1 = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
clf2 = XGBClassifier(random_state=42, n_estimators=200, max_depth=8, learning_rate=0.05, subsample=0.8)
clf3 = SVC(probability=True, random_state=42, C=10, kernel='rbf', gamma=0.1)
clf4 = LogisticRegression(random_state=42)
clf5 = KNeighborsClassifier(n_neighbors=5)
clf6 = GradientBoostingClassifier(n_estimators=100, random_state=42)

# 9. 使用 VotingClassifier 进行集成学习
eclf = VotingClassifier(
    estimators=[('rf', clf1), ('xgb', clf2), ('svc', clf3), ('lr', clf4), ('knn', clf5), ('gb', clf6)],
    voting='soft',
    weights=[1, 2, 1, 1, 1, 2]  # 根据模型表现分配权重，可以通过实验调整
)

# 10. 定义超参数搜索的参数网格
param_grid = {
    'xgb__n_estimators': [100, 200],
    'xgb__max_depth': [6, 8],
    'xgb__learning_rate': [0.01, 0.05],
    'svc__C': [1, 10],
    'svc__gamma': [0.01, 0.1],
    'knn__n_neighbors': [3, 5],
    'gb__n_estimators': [100, 200],
    'gb__learning_rate': [0.01, 0.1]
}

# 11. 使用 GridSearchCV 进行超参数搜索
grid_search = GridSearchCV(eclf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(X_train_res, y_train_res)

# 输出最佳参数
print(f"最佳参数: {grid_search.best_params_}")

# 使用最佳参数的模型进行预测
y_pred = grid_search.predict(X_test_scaled)
y_test_labels = label_encoder.inverse_transform(y_test)  # 转回原始类别名称
y_pred_labels = label_encoder.inverse_transform(y_pred)

# 12. 输出分类报告
print(classification_report(y_test_labels, y_pred_labels))

# 13. 保存训练好的模型
model_filename = 'voting_classifier_model_with_gridsearch.pkl'
joblib.dump(grid_search.best_estimator_, model_filename)
print(f"集成模型已保存为 {model_filename}")

# 14. 保存归一化模型和标签编码器
scaler_filename = 'scaler.pkl'
joblib.dump(scaler, scaler_filename)

label_encoder_filename = 'label_encoder.pkl'
joblib.dump(label_encoder, label_encoder_filename)
print(f"归一化模型和标签编码器已保存")
