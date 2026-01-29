# %%
import time
# 全局变量存储开始时间（类似MATLAB的工作区变量）
_tic_toc_start_time = None
def tic():
    """启动计时器（对应MATLAB tic）"""
    global _tic_toc_start_time
    _tic_toc_start_time = time.perf_counter()  # 高精度计时器（推荐，比time.time()准）

def toc(print_msg="Elapsed time: "):
    """结束计时器并打印耗时（对应MATLAB toc）
    参数：print_msg - 自定义提示语
    返回：耗时（秒）
    """
    if _tic_toc_start_time is None:
        print("请先调用tic()启动计时器！")
        return None
    elapsed_time = time.perf_counter() - _tic_toc_start_time
    print(f"{print_msg}{elapsed_time:.4f} seconds")
    return elapsed_time  # 可选：返回耗时数值，方便后续计算

# %%
tic()  # 启动计时器
# 模拟一些耗时操作

# %%
# 数值计算与随机数
import numpy as np

# 表格数据处理（本例用于把 X/y 包成 DataFrame，方便分组画图）
import pandas as pd

# 画图
import matplotlib.pyplot as plt

# 标准化：把不同量纲/范围的特征缩放到均值0、方差1附近
# 对 SVM / RBF 这类距离敏感的模型非常重要
from sklearn.preprocessing import StandardScaler

# 支持向量机分类器
from sklearn.svm import SVC

# Pipeline：把“标准化 + 模型”串起来，避免数据泄漏（只在训练集 fit scaler）
from sklearn.pipeline import Pipeline

# 网格搜索 + 交叉验证
from sklearn.model_selection import GridSearchCV

# 模型评估指标：准确率
from sklearn.metrics import accuracy_score


# %%
def make_dataset(n_per_class=200, seed=None):
    """
    生成一个二分类数据集（2D 特征）：
    特征为 (x, y)，其中 y 由一个二次函数 + 噪声构成。

    公式基底：
        y = 2 + 3x + x^2 + noise

    两个类别的区别：
        class 0: noise = + 10 * U(0,1)  （往上偏）
        class 1: noise = - 10 * U(0,1)  （往下偏）

    参数
    ----
    n_per_class : int
        每个类别的样本数，总样本数 = 2 * n_per_class
    seed : int or None
        随机种子，用于可复现实验

    返回
    ----
    df : pandas.DataFrame
        列：
            x1: 第一维特征（这里其实就是 x）
            x2: 第二维特征（这里其实就是 y）
            label: 0/1 标签
    """
    # 推荐用 numpy 的新随机数生成器，保证可复现且更现代
    rng = np.random.default_rng(seed)

    # ===== 类别 0 =====
    # x 在 [-5, 5] 均匀采样
    x1 = -5 + 10 * rng.random(n_per_class)

    # y = 二次函数 + 向上噪声（+ 10*U(0,1)）
    y1 = 2 + 3 * x1 + 1 * (x1 ** 2) + 10 * rng.random(n_per_class)

    # ===== 类别 1 =====
    x2 = -5 + 10 * rng.random(n_per_class)

    # y = 二次函数 + 向下噪声（- 10*U(0,1)）
    y2 = 2 + 3 * x2 + 1 * (x2 ** 2) - 10 * rng.random(n_per_class)

    # 把两类拼在一起，形成 (N, 2) 的特征矩阵
    X = np.vstack([
        np.column_stack([x1, y1]),
        np.column_stack([x2, y2]),
    ])

    # 标签：前 n_per_class 个为 0，后 n_per_class 个为 1
    y = np.array([0] * n_per_class + [1] * n_per_class)

    # 用 DataFrame 方便后续画图时按 label/pred 分组
    df = pd.DataFrame(X, columns=["x1", "x2"])
    df["label"] = y.astype(int)
    return df


# %%
def plot_scatter(df, color_col, title):
    """
    按 df[color_col] 分组上色画散点图

    参数
    ----
    df : DataFrame
        必须包含 x1, x2 两列
    color_col : str
        用于分组/上色的列名（如 'label' 或 'pred'）
    title : str
        图标题
    """
    plt.figure(figsize=(6, 5))

    # groupby 后，每个 k 是类别值（0/1），sub 是对应子表
    for k, sub in df.groupby(color_col):
        plt.scatter(sub["x1"], sub["x2"], s=18, alpha=0.8, label=str(k))

    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend(title=color_col)
    plt.tight_layout()
    plt.show()


# %%
# 生成训练集（seed=42 固定随机性，保证每次运行一致）
train_df = make_dataset(n_per_class=200, seed=42)

# 生成测试集（使用不同 seed，保证测试数据与训练不完全相同）
test_df  = make_dataset(n_per_class=200, seed=202)

# 先看看真实标签下的分布
plot_scatter(train_df, "label", "Training Data (raw)")
plot_scatter(test_df,  "label", "Testing Data (raw)")

# ===== 准备给 sklearn 的输入 =====
# X: 形状 (N, 2)
# y: 形状 (N,)
X_train = train_df[["x1", "x2"]].to_numpy()
y_train = train_df["label"].to_numpy()

X_test = test_df[["x1", "x2"]].to_numpy()
y_test = test_df["label"].to_numpy()


# %%
# Pipeline 的意义：
# 1) scaler 只在训练集 fit（避免数据泄漏：不能拿测试集信息参与标准化参数估计）
# 2) predict 时会自动对测试集做 transform，然后再送进 SVC
base_clf = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("svc", SVC(kernel="rbf", gamma=0.01))
])

# 在训练集上训练
base_clf.fit(X_train, y_train)

# 在测试集上预测
pred = base_clf.predict(X_test)

# 测试集准确率
acc = accuracy_score(y_test, pred)

# 把预测结果加回 test_df，方便按 pred 上色画图
test_vis = test_df.copy()
test_vis["pred"] = pred

plot_scatter(test_vis, "pred", f"RBF kernel (gamma=0.01) - Accuracy={acc:.4f}")

# Notebook 里直接显示参数字典（比 print 更舒服）
base_clf.get_params()


# %%
# 手动给一组 gamma，观察模型表现与决策复杂度的变化
gammas = [0.01, 0.1, 1, 10, 100, 1000]

for g in gammas:
    # 每次重新建一个 pipeline（不要复用同一个对象，避免上一次 fit 的状态干扰）
    clf = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", gamma=g))
    ])

    # 训练
    clf.fit(X_train, y_train)

    # 预测与评估
    pred_g = clf.predict(X_test)
    acc_g = accuracy_score(y_test, pred_g)

    # 画出预测分区效果（用 pred 上色）
    test_vis = test_df.copy()
    test_vis["pred"] = pred_g
    plot_scatter(test_vis, "pred", f"Gamma={g}  Accuracy={acc_g:.4f}")


# %%
# 注意：这里把 gamma 和 C 交给 GridSearchCV 搜索
# C：惩罚系数（越大越倾向于把训练数据分得更“干净”）
# gamma：RBF 核参数（越大边界越复杂）
pipe = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("svc", SVC(kernel="rbf"))
])

# 参数网格：
# - gamma: 0.0 到 10.0 步长 0.1（注意：gamma=0 通常没意义，但保留是为了和你原逻辑一致）
# - C: 2^1 到 2^5
param_grid = {
    "svc__gamma": np.arange(0.0, 10.0 + 1e-9, 0.1),
    "svc__C": [2**i for i in range(1, 6)],
}

# 5 折交叉验证：
# 把训练集分成 5 份，每次用 4 份训练、1 份验证，循环 5 次取平均准确率
# 这样能减少“碰巧分对/分错”导致的参数选择偏差
gs = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring="accuracy",
    cv=5,
    n_jobs=-1
)

# 只用训练集做网格搜索（测试集必须留到最后，只做一次最终评估）
gs.fit(X_train, y_train)

# 输出：最好的 CV 分数与参数组合
gs.best_score_, gs.best_params_


# %%
# best_estimator_ 是已经在“整个训练集”上重新训练过的最佳 pipeline
best_model = gs.best_estimator_

# 用最佳模型在测试集上预测
best_pred = best_model.predict(X_test)

# 测试集准确率（这才是最终泛化效果）
best_acc = accuracy_score(y_test, best_pred)

print("Best CV score:", gs.best_score_)
print("Best params:", gs.best_params_)
print("Test accuracy with best model:", best_acc)

# 画预测结果
test_vis = test_df.copy()
test_vis["pred"] = best_pred
plot_scatter(test_vis, "pred", f"Best RBF Model - Accuracy={best_acc:.4f}")

# %%
toc("Total elapsed time: ")


