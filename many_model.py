import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

# 读取数据
similarity_data = pd.read_csv('similarity_results.csv')

# 准备特征矩阵X和标签y
X = similarity_data.drop(columns=['image_name', 'true_class']).applymap(lambda x: float(x.strip('[]')))
y = similarity_data['true_class']

# 对标签进行编码
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# 存储模型的字典
models = {
    "SVM": SVC(kernel='linear'),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Gradient Boosting": GradientBoostingClassifier(),
    "MLP": MLPClassifier(max_iter=500)
}

# 存储分类报告的字典
reports = {}

# 训练并评估每个模型
for model_name, model in models.items():
    # 训练模型
    model.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = model.predict(X_test)

    # 生成分类报告
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    reports[model_name] = report

# 打印每个模型的分类报告
for model_name, report in reports.items():
    print(f"=== {model_name} ===")
    print(report)
