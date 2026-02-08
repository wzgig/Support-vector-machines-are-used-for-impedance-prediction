# %%
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =============================================================================
# 计时器：tic / toc（模仿 MATLAB）
# =============================================================================
_tic_toc_start_time = None

def tic():
    """启动计时器（对应 MATLAB tic）"""
    global _tic_toc_start_time
    _tic_toc_start_time = time.perf_counter()

def toc(print_msg="Elapsed time: "):
    """结束计时器并打印耗时（对应 MATLAB toc）"""
    if _tic_toc_start_time is None:
        print("请先调用 tic() 启动计时器！")
        return None
    elapsed_time = time.perf_counter() - _tic_toc_start_time
    print(f"{print_msg}{elapsed_time:.4f} seconds")
    return elapsed_time


# =============================================================================
# 数据生成：一元回归 y = 2 + 3x + x^2 + noise
# =============================================================================
def make_regression_dataset(n=600, seed=42, noise_std=4.0):
    """
    生成一元回归数据：
        y = 2 + 3x + x^2 + noise

    参数
    ----
    n : 样本数
    seed : 随机种子（保证可复现）
    noise_std : 噪声标准差（越大越难拟合）

    返回
    ----
    df : DataFrame, columns=['x','y']
    """
    rng = np.random.default_rng(seed)
    x = -5 + 10 * rng.random(n)               # x ~ Uniform[-5, 5]
    noise = rng.normal(0.0, noise_std, n)     # noise ~ N(0, noise_std^2)
    y = 2 + 3 * x + (x ** 2) + noise
    return pd.DataFrame({"x": x, "y": y})


def true_function(x):
    """无噪声真实函数（用于画参考曲线）"""
    return 2 + 3 * x + x**2


# =============================================================================
# 绘图函数
# =============================================================================
def plot_regression_scatter(train_df, test_df, title="Regression data"):
    """画训练/测试散点，直观看数据规律与噪声"""
    plt.figure(figsize=(7, 5))
    plt.scatter(train_df["x"], train_df["y"], s=18, alpha=0.7, label="train")
    plt.scatter(test_df["x"],  test_df["y"],  s=18, alpha=0.7, label="test")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_model_curve(model, train_df, test_df, title, show_truth=True):
    """
    一元回归最直观的可视化：散点 + 模型预测曲线
    - 在 x 的网格上预测 y_hat，画出平滑曲线
    - 可选：叠加真实无噪声函数曲线，便于对比模型学得如何
    """
    x_min = min(train_df["x"].min(), test_df["x"].min())
    x_max = max(train_df["x"].max(), test_df["x"].max())

    # 生成稠密网格用于画曲线（注意：sklearn 需要二维输入）
    x_grid = np.linspace(x_min, x_max, 400).reshape(-1, 1)
    y_hat = model.predict(x_grid)

    plt.figure(figsize=(7, 5))
    plt.scatter(train_df["x"], train_df["y"], s=18, alpha=0.55, label="train")
    plt.scatter(test_df["x"],  test_df["y"],  s=18, alpha=0.55, label="test")
    plt.plot(x_grid.ravel(), y_hat, linewidth=2, label="SVR prediction")

    if show_truth:
        plt.plot(x_grid.ravel(), true_function(x_grid.ravel()),
                 linewidth=2, linestyle="--", label="True function (no noise)")

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# =============================================================================
# 评估报告
# =============================================================================
def regression_report(y_true, y_pred, prefix=""):
    """
    输出回归指标：
    - MAE：平均绝对误差（越小越好）
    - RMSE：均方根误差（越小越好，对大误差更敏感）
    - R2：拟合优度（越接近 1 越好；可能为负，表示比常数预测还差）
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print(f"{prefix}MAE  = {mae:.4f}")
    print(f"{prefix}RMSE = {rmse:.4f}")
    print(f"{prefix}R2   = {r2:.4f}")
    return {"mae": mae, "rmse": rmse, "r2": r2}


# =============================================================================
# 主流程
# =============================================================================
tic()

# 1) 生成数据，并做一次标准 train/test 切分（更常见、更可复现实验对比）
df = make_regression_dataset(n=600, seed=42, noise_std=4.0)

train_df, test_df = train_test_split(
    df, test_size=0.5, random_state=202, shuffle=True
)

plot_regression_scatter(train_df, test_df, title="SVR Regression Data (raw)")

X_train = train_df[["x"]].to_numpy()
y_train = train_df["y"].to_numpy()
X_test  = test_df[["x"]].to_numpy()
y_test  = test_df["y"].to_numpy()


# 2) Baseline：RBF-SVR + 标准化
# 为什么必须 StandardScaler？
# - RBF 核依赖样本间距离；如果特征尺度不合适，gamma/C 的意义会变得很不稳定
base_model = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("svr", SVR(kernel="rbf", C=10.0, gamma=0.5, epsilon=0.5)),
])

base_model.fit(X_train, y_train)
base_pred = base_model.predict(X_test)

print("=== Baseline SVR on test set ===")
regression_report(y_test, base_pred)
plot_model_curve(base_model, train_df, test_df, title="Baseline SVR (RBF)")


# 3) 手动观察 gamma 对拟合复杂度的影响（可选批量画图）
# gamma 小 -> 核更“宽”，更平滑，易欠拟合
# gamma 大 -> 核更“窄”，更抖动，易过拟合
SHOW_GAMMA_PLOTS = True
gammas = [0.01, 0.05, 0.1, 0.5, 1, 5]

for g in gammas:
    model = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="rbf", C=10.0, gamma=g, epsilon=0.5)),
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print(f"\n=== gamma={g} ===")
    regression_report(y_test, pred, prefix="  ")

    if SHOW_GAMMA_PLOTS:
        plot_model_curve(model, train_df, test_df, title=f"SVR (gamma={g})")


# 4) GridSearchCV：同时搜索 C / gamma / epsilon
# - C：惩罚系数，大 -> 更“严格”，更贴近训练集（复杂度更高）
# - gamma：核宽度参数，大 -> 更局部、更复杂（更易过拟合）
# - epsilon：ε-insensitive tube 的宽度，大 -> 容忍更多误差不计入损失（更平滑）
pipe = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("svr", SVR(kernel="rbf")),
])

param_grid = {
    "svr__C":       [1, 3, 10, 30, 100],
    "svr__gamma":   [0.01, 0.03, 0.1, 0.3, 1, 3],
    "svr__epsilon": [0.1, 0.3, 0.5, 1.0],
}

# 回归常用评分：neg_mean_squared_error / neg_mean_absolute_error / r2
# 这里仍用 neg_mean_squared_error（注意：越大越好，因为是负号）
cv = KFold(n_splits=5, shuffle=True, random_state=42)

gs = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring="neg_mean_squared_error",
    cv=cv,
    n_jobs=-1,
    refit=True,   # 用最优参数在全训练集上重新训练 best_estimator_
)

gs.fit(X_train, y_train)

best_neg_mse = gs.best_score_
best_rmse_cv = np.sqrt(-best_neg_mse)

print("\n=== GridSearch best result (CV) ===")
print(f"Best CV RMSE: {best_rmse_cv:.4f} (from best neg-MSE={best_neg_mse:.6f})")
print("Best params:", gs.best_params_)


# 5) 最佳模型在测试集做最终评估（测试集只用一次，避免调参看测试集导致偏乐观）
best_model = gs.best_estimator_
best_pred = best_model.predict(X_test)

print("\n=== Best SVR on test set ===")
best_metrics = regression_report(y_test, best_pred)

plot_model_curve(
    best_model,
    train_df,
    test_df,
    title=f"Best SVR (MAE={best_metrics['mae']:.3f}, RMSE={best_metrics['rmse']:.3f}, R2={best_metrics['r2']:.3f})",
)


# 6) 预测 vs 真实：理想情况点落在 y=x 对角线附近
plt.figure(figsize=(6, 6))
plt.scatter(y_test, best_pred, s=18, alpha=0.7)
minv = min(y_test.min(), best_pred.min())
maxv = max(y_test.max(), best_pred.max())
plt.plot([minv, maxv], [minv, maxv], linewidth=2)
plt.title("Predicted vs True (Best SVR)")
plt.xlabel("True y")
plt.ylabel("Predicted y")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# 7) 残差分布：残差 = y_true - y_pred
residual = y_test - best_pred
plt.figure(figsize=(7, 4))
plt.hist(residual, bins=30, alpha=0.8)
plt.title("Residual histogram (y_true - y_pred)")
plt.xlabel("Residual")
plt.ylabel("Count")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

toc("Total elapsed time: ")
