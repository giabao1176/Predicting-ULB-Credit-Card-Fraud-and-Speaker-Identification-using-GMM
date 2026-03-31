"""
=============================================================================
  PHÁT HIỆN BẤT THƯỜNG TRONG TÀI CHÍNH - CREDIT CARD FRAUD DETECTION
  Sử dụng: Gaussian Mixture Model (GMM) & So sánh với K-Means
  Tập dữ liệu: ULB Machine Learning Group - creditcard.csv
=============================================================================

CẤU TRÚC FILE CSV:
------------------
 - Time   : Số giây trôi qua kể từ giao dịch đầu tiên trong bộ dữ liệu
 - V1~V28 : 28 thành phần chính (PCA) - đã được ẩn danh hoá để bảo vệ quyền riêng tư
 - Amount : Số tiền giao dịch (USD)
 - Class  : Nhãn thật (0 = bình thường, 1 = gian lận)

Tập dữ liệu cực kỳ mất cân bằng:
 - ~284,315 giao dịch bình thường (99.83%)
 - ~492 giao dịch gian lận (0.17%)

LƯU Ý: GMM được huấn luyện KHÔNG DÙNG nhãn Class (unsupervised).
        Nhãn chỉ dùng để đánh giá hiệu năng mô hình ở cuối.
=============================================================================
"""

# ─────────────────────────────────────────────────────────────
# BƯỚC 0: Import thư viện
# ─────────────────────────────────────────────────────────────
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (chạy không cần GUI)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
    ConfusionMatrixDisplay
)
from scipy.stats import gaussian_kde
import os

# ─────────────────────────────────────────────────────────────
# CẤU HÌNH
# ─────────────────────────────────────────────────────────────
DATA_PATH   = r"d:\HocTriTueNhanTao\math\creditcard.csv"
OUT_DIR     = r"d:\HocTriTueNhanTao\math\plots"
SAMPLE_SIZE = 50_000   # Lấy mẫu để tăng tốc (None = toàn bộ)
GMM_COMPONENTS = 5     # Số Gaussian trong hỗn hợp
KMEANS_K    = 2        # K cho K-Means
THRESHOLD_PERCENTILE = 2  # Ngưỡng: % thấp nhất của log-likelihood => dị thường
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

os.makedirs(OUT_DIR, exist_ok=True)

def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Đã lưu: {path}")

print("=" * 65)
print("  CREDIT CARD FRAUD DETECTION - ANOMALY DETECTION PIPELINE")
print("=" * 65)

# ─────────────────────────────────────────────────────────────
# BƯỚC 1: ĐỌC & KHÁM PHÁ DỮ LIỆU (EDA)
# ─────────────────────────────────────────────────────────────
print("\n[BƯỚC 1] Đọc và khám phá dữ liệu...")
df = pd.read_csv(DATA_PATH)

print(f"  Kích thước bộ dữ liệu : {df.shape[0]:,} dòng × {df.shape[1]} cột")
print(f"  Giá trị null          : {df.isnull().sum().sum()}")
print(f"\n  Phân phối nhãn (Class):")
vc = df['Class'].value_counts()
for label, cnt in vc.items():
    tag = "Bình thường" if label == 0 else "GianLận    "
    print(f"    Class={label} [{tag}]: {cnt:>7,}  ({cnt/len(df)*100:.3f}%)")

# ── Biểu đồ 1: Tổng quan bộ dữ liệu ──────────────────────────
fig = plt.figure(figsize=(16, 10))
fig.suptitle("TỔNG QUAN BỘ DỮ LIỆU - CREDIT CARD FRAUD DETECTION\n"
             "(Nghiên cứu của Nhóm Machine Learning - ĐH ULB, Bỉ)",
             fontsize=14, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# Pie chart phân phối nhãn
ax0 = fig.add_subplot(gs[0, 0])
colors_pie = ['#2196F3', '#F44336']
wedges, texts, autotexts = ax0.pie(
    vc.values, labels=['Bình thường\n(Class=0)', 'Gian lận\n(Class=1)'],
    autopct='%1.3f%%', colors=colors_pie, startangle=140,
    wedgeprops=dict(edgecolor='white', linewidth=2))
for at in autotexts: at.set_fontsize(9)
ax0.set_title("Phân phối nhãn\n(Cực kỳ mất cân bằng!)", fontsize=11)

# Phân phối Time
ax1 = fig.add_subplot(gs[0, 1])
ax1.hist(df[df['Class']==0]['Time']/3600, bins=60, alpha=0.7,
         color='#2196F3', label='Bình thường', density=True)
ax1.hist(df[df['Class']==1]['Time']/3600, bins=60, alpha=0.8,
         color='#F44336', label='Gian lận', density=True)
ax1.set_xlabel("Thời gian (giờ)", fontsize=10)
ax1.set_ylabel("Mật độ", fontsize=10)
ax1.set_title("Phân phối theo thời gian\n(giờ kể từ GD đầu tiên)", fontsize=11)
ax1.legend(fontsize=9)

# Phân phối Amount
ax2 = fig.add_subplot(gs[0, 2])
ax2.hist(df[df['Class']==0]['Amount'].clip(0, 500), bins=80, alpha=0.7,
         color='#2196F3', label='Bình thường', density=True)
ax2.hist(df[df['Class']==1]['Amount'].clip(0, 500), bins=80, alpha=0.8,
         color='#F44336', label='Gian lận', density=True)
ax2.set_xlabel("Số tiền GD (USD, cắt tại 500)", fontsize=10)
ax2.set_ylabel("Mật độ", fontsize=10)
ax2.set_title("Phân phối số tiền giao dịch", fontsize=11)
ax2.legend(fontsize=9)

# Box-plot Amount
ax3 = fig.add_subplot(gs[1, 0])
data_bp = [df[df['Class']==0]['Amount'].values,
           df[df['Class']==1]['Amount'].values]
bp = ax3.boxplot(data_bp, labels=['Bình thường', 'Gian lận'],
                 patch_artist=True, notch=False)
for patch, color in zip(bp['boxes'], ['#2196F3', '#F44336']):
    patch.set_facecolor(color); patch.set_alpha(0.7)
ax3.set_ylabel("Số tiền (USD)", fontsize=10)
ax3.set_title("Box-plot số tiền theo nhãn", fontsize=11)
ax3.set_ylim(-50, 800)

# Heatmap tương quan (top 10 features)
ax4 = fig.add_subplot(gs[1, 1:])
feats = ['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','Amount']
corr = df[feats + ['Class']].corr()['Class'].drop('Class').sort_values()
colors_bar = ['#F44336' if c < 0 else '#2196F3' for c in corr]
ax4.barh(range(len(corr)), corr.values, color=colors_bar, edgecolor='white')
ax4.set_yticks(range(len(corr)))
ax4.set_yticklabels(corr.index, fontsize=9)
ax4.axvline(0, color='black', linewidth=0.8)
ax4.set_xlabel("Hệ số tương quan với Class (1=Gian lận)", fontsize=10)
ax4.set_title("Tương quan biến với nhãn Class\n(đỏ = tương quan âm, xanh = dương)", fontsize=11)

fig.text(0.5, 0.01,
         "Bộ dữ liệu gồm 284,807 giao dịch | 28 đặc trưng PCA (V1-V28) | Amount | Time | Class",
         ha='center', fontsize=9, color='gray')
save(fig, "01_dataset_overview.png")

# ─────────────────────────────────────────────────────────────
# BƯỚC 2: TIỀN XỬ LÝ
# ─────────────────────────────────────────────────────────────
print("\n[BƯỚC 2] Tiền xử lý dữ liệu...")

# Lấy mẫu nếu cần
if SAMPLE_SIZE and len(df) > SAMPLE_SIZE:
    # Giữ nguyên tỷ lệ lớp
    n_fraud  = len(df[df['Class']==1])
    n_normal = SAMPLE_SIZE - n_fraud
    df_sample = pd.concat([
        df[df['Class']==0].sample(n_normal, random_state=RANDOM_SEED),
        df[df['Class']==1]
    ]).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    print(f"  Đã lấy mẫu: {len(df_sample):,} dòng (giữ toàn bộ {n_fraud} gian lận)")
else:
    df_sample = df.copy()
    print(f"  Dùng toàn bộ: {len(df_sample):,} dòng")

y_true = df_sample['Class'].values
features = [f'V{i}' for i in range(1, 29)] + ['Amount']

# Chuẩn hoá
scaler = StandardScaler()
X = scaler.fit_transform(df_sample[features])
print(f"  Số đặc trưng sau chuẩn hoá: {X.shape[1]}")

# Tách tập "bình thường" để train GMM (unsupervised: không dùng nhãn khi predict)
X_normal = X[y_true == 0]
print(f"  GMM sẽ huấn luyện trên {X_normal.shape[0]:,} giao dịch bình thường")

# Giảm chiều bằng PCA (cho visualisation)
pca = PCA(n_components=2, random_state=RANDOM_SEED)
X_pca = pca.fit_transform(X)
print(f"  PCA 2D: phương sai giải thích = "
      f"{sum(pca.explained_variance_ratio_)*100:.1f}%")

# ─────────────────────────────────────────────────────────────
# BƯỚC 3: GAUSSIAN MIXTURE MODEL (GMM)
# ─────────────────────────────────────────────────────────────
print(f"\n[BƯỚC 3] Huấn luyện Gaussian Mixture Model (K={GMM_COMPONENTS})...")
print("  GMM học 'hình dạng' của giao dịch bình thường bằng cách ước lượng")
print("  hỗn hợp các phân phối Gaussian đa biến.")

gmm = GaussianMixture(
    n_components=GMM_COMPONENTS,
    covariance_type='full',
    max_iter=200,
    n_init=3,
    random_state=RANDOM_SEED
)
gmm.fit(X_normal)

# Chọn số component tối ưu bằng BIC
print("\n  Lựa chọn K tối ưu bằng BIC (Bayesian Information Criterion):")
bic_scores = []
k_range = range(1, 11)
for k in k_range:
    g = GaussianMixture(n_components=k, covariance_type='full',
                        max_iter=100, random_state=RANDOM_SEED)
    g.fit(X_normal)
    bic_scores.append(g.bic(X_normal))
    print(f"    K={k}: BIC = {bic_scores[-1]:.0f}")

# Tính log-likelihood (score) cho toàn bộ tập
log_probs = gmm.score_samples(X)  # mật độ xác suất log cho mỗi điểm

# Ngưỡng phát hiện: ngưỡng dưới percentile_thresh
threshold = np.percentile(log_probs, THRESHOLD_PERCENTILE)
print(f"\n  Ngưỡng log-likelihood (percentile {THRESHOLD_PERCENTILE}%): {threshold:.4f}")
print("  → Giao dịch có log p(x) < ngưỡng sẽ bị gắn cờ là 'nghi ngờ gian lận'")

y_pred_gmm = (log_probs < threshold).astype(int)

# ── Biểu đồ 2: GMM - BIC & Log-Likelihood ─────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("BƯỚC 3 - GAUSSIAN MIXTURE MODEL: Chọn K & Phân phối Log-Likelihood",
             fontsize=13, fontweight='bold')

# BIC
ax = axes[0]
ax.plot(list(k_range), bic_scores, 'o-', color='#673AB7', linewidth=2, markersize=8)
ax.fill_between(list(k_range), bic_scores, alpha=0.15, color='#673AB7')
best_k = list(k_range)[np.argmin(bic_scores)]
ax.axvline(best_k, color='red', linestyle='--', label=f'Tốt nhất K={best_k}')
ax.axvline(GMM_COMPONENTS, color='green', linestyle=':', linewidth=2,
           label=f'Dùng K={GMM_COMPONENTS}')
ax.set_xlabel("Số thành phần Gaussian (K)", fontsize=11)
ax.set_ylabel("BIC (càng thấp càng tốt)", fontsize=11)
ax.set_title("Lựa chọn K bằng BIC\n(Bayesian Information Criterion)", fontsize=11)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Histogram log-likelihood
ax = axes[1]
lp_normal = log_probs[y_true == 0]
lp_fraud  = log_probs[y_true == 1]
ax.hist(lp_normal, bins=100, density=True, alpha=0.6, color='#2196F3',
        label='Bình thường', range=(-500, 50))
ax.hist(lp_fraud,  bins=100, density=True, alpha=0.8, color='#F44336',
        label='Gian lận', range=(-500, 50))
ax.axvline(threshold, color='black', linestyle='--', linewidth=2,
           label=f'Ngưỡng={threshold:.1f}')
ax.set_xlabel("Log-likelihood  log p(x|GMM)", fontsize=11)
ax.set_ylabel("Mật độ", fontsize=11)
ax.set_title(f"Phân phối Log-Likelihood\n(ngưỡng = percentile {THRESHOLD_PERCENTILE}%)", fontsize=11)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Không gian PCA tô màu theo dự đoán GMM
ax = axes[2]
colors_pred = ['#2196F3' if p==0 else '#F44336' for p in y_pred_gmm]
ax.scatter(X_pca[y_pred_gmm==0, 0], X_pca[y_pred_gmm==0, 1],
           c='#2196F3', s=5, alpha=0.3, label='Bình thường')
ax.scatter(X_pca[y_pred_gmm==1, 0], X_pca[y_pred_gmm==1, 1],
           c='#F44336', s=30, alpha=0.7, label='Nghi ngờ gian lận', zorder=5)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=11)
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=11)
ax.set_title("Không gian 2D (PCA)\nCác điểm được GMM gắn cờ", fontsize=11)
ax.legend(fontsize=10, markerscale=3)
ax.grid(alpha=0.3)

save(fig, "02_gmm_training.png")

# ── Biểu đồ 3: Visualize GMM components trong PCA 2D ──────────
print("\n[BƯỚC 3b] Visualize các thành phần Gaussian trong không gian 2D...")
gmm_2d = GaussianMixture(n_components=GMM_COMPONENTS, covariance_type='full',
                          random_state=RANDOM_SEED)
X_normal_2d = X_pca[y_true == 0]
gmm_2d.fit(X_normal_2d)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("BƯỚC 3b - VISUALIZE CÁC THÀNH PHẦN GAUSSIAN TRONG KHÔNG GIAN PCA 2D",
             fontsize=13, fontweight='bold')

for ax, show_fraud in zip(axes, [False, True]):
    # Tạo lưới
    x1 = np.linspace(X_pca[:,0].min()-1, X_pca[:,0].max()+1, 200)
    x2 = np.linspace(X_pca[:,1].min()-1, X_pca[:,1].max()+1, 200)
    xx, yy = np.meshgrid(x1, x2)
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = gmm_2d.score_samples(grid).reshape(xx.shape)

    cf = ax.contourf(xx, yy, zz, levels=30, cmap='Blues', alpha=0.7)
    ax.contour(xx, yy, zz, levels=15, colors='white', alpha=0.3, linewidths=0.5)
    plt.colorbar(cf, ax=ax, label='Log-Likelihood')

    # Scatter
    ax.scatter(X_pca[y_true==0, 0], X_pca[y_true==0, 1],
               c='#4CAF50', s=4, alpha=0.2, label='Bình thường')
    if show_fraud:
        ax.scatter(X_pca[y_true==1, 0], X_pca[y_true==1, 1],
                   c='#F44336', s=60, marker='*', alpha=0.9,
                   label='Gian lận (thật)', zorder=10)
    # Vẽ tâm Gaussian
    means2d = gmm_2d.means_
    for i, m in enumerate(means2d):
        ax.scatter(*m, c='yellow', s=200, marker='X', edgecolors='black',
                   linewidths=1.5, zorder=11, label=f'Tâm G{i+1}' if i==0 else "")

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=11)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=11)
    title = "Mật độ GMM - chỉ dữ liệu bình thường" if not show_fraud \
            else "Mật độ GMM - với điểm gian lận thật (★)"
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9, markerscale=2)

save(fig, "03_gmm_components_2d.png")

# ─────────────────────────────────────────────────────────────
# BƯỚC 4: K-MEANS - SO SÁNH
# ─────────────────────────────────────────────────────────────
print(f"\n[BƯỚC 4] Huấn luyện K-Means (K={KMEANS_K}) để so sánh...")
print("  K-Means phân cụm cứng dựa vào khoảng cách Euclidean đến tâm cụm.")
print("  Phát hiện bất thường: điểm nào xa tâm cụm gần nhất quá ngưỡng => dị thường")

kmeans = KMeans(n_clusters=KMEANS_K, n_init=10, random_state=RANDOM_SEED)
kmeans.fit(X)

# Khoảng cách đến tâm cụm gần nhất
distances = np.min(
    np.linalg.norm(X[:, np.newaxis, :] - kmeans.cluster_centers_[np.newaxis, :, :], axis=2),
    axis=1
)
dist_threshold = np.percentile(distances, 100 - THRESHOLD_PERCENTILE)
y_pred_km = (distances > dist_threshold).astype(int)
print(f"  Ngưỡng khoảng cách (percentile {100-THRESHOLD_PERCENTILE}%): {dist_threshold:.4f}")

# ── Biểu đồ 4: K-Means chi tiết ───────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("BƯỚC 4 - K-MEANS CLUSTERING: Phân cụm & Phát hiện bất thường",
             fontsize=13, fontweight='bold')

cluster_colors = ['#03A9F4', '#FF9800']
ax = axes[0]
for k in range(KMEANS_K):
    mask = kmeans.labels_ == k
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], s=5, alpha=0.3,
               color=cluster_colors[k], label=f'Cụm {k}')
centers_2d = pca.transform(kmeans.cluster_centers_)
ax.scatter(centers_2d[:, 0], centers_2d[:, 1], c='red', s=250,
           marker='X', edgecolors='black', linewidths=1.5, zorder=10, label='Tâm cụm')
ax.set_title("K-Means Clustering (K=2)\ntrong không gian PCA 2D", fontsize=11)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=11)
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=11)
ax.legend(fontsize=10, markerscale=3)
ax.grid(alpha=0.3)

ax = axes[1]
ax.hist(distances[y_true==0], bins=100, density=True, alpha=0.6,
        color='#2196F3', label='Bình thường', range=(0, distances.max()))
ax.hist(distances[y_true==1], bins=100, density=True, alpha=0.8,
        color='#F44336', label='Gian lận', range=(0, distances.max()))
ax.axvline(dist_threshold, color='black', linestyle='--', linewidth=2,
           label=f'Ngưỡng = {dist_threshold:.2f}')
ax.set_xlabel("Khoảng cách đến tâm cụm gần nhất", fontsize=11)
ax.set_ylabel("Mật độ", fontsize=11)
ax.set_title("Phân phối khoảng cách K-Means\n(điểm xa = bất thường)", fontsize=11)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

ax = axes[2]
ax.scatter(X_pca[y_pred_km==0, 0], X_pca[y_pred_km==0, 1],
           c='#2196F3', s=5, alpha=0.3, label='Bình thường')
ax.scatter(X_pca[y_pred_km==1, 0], X_pca[y_pred_km==1, 1],
           c='#F44336', s=30, alpha=0.7, label='Nghi ngờ', zorder=5)
ax.set_title("Các điểm bị K-Means gắn cờ\n(không gian PCA)", fontsize=11)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=11)
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=11)
ax.legend(fontsize=10, markerscale=3)
ax.grid(alpha=0.3)

save(fig, "04_kmeans_analysis.png")

# ─────────────────────────────────────────────────────────────
# BƯỚC 5: ĐÁNH GIÁ & SO SÁNH
# ─────────────────────────────────────────────────────────────
print("\n[BƯỚC 5] Đánh giá và so sánh GMM vs K-Means...")

def evaluate(name, y_true, y_pred, scores):
    """In báo cáo và trả về dict kết quả"""
    print(f"\n  ── {name} ──")
    print(classification_report(y_true, y_pred,
          target_names=['Bình thường', 'Gian lận']))
    auc = roc_auc_score(y_true, scores)
    ap  = average_precision_score(y_true, scores)
    print(f"  ROC-AUC : {auc:.4f}")
    print(f"  Average Precision (AP): {ap:.4f}")
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"  TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    return dict(name=name, auc=auc, ap=ap, cm=cm,
                y_pred=y_pred, scores=scores)

# Scores: GMM dùng -log_probs (cao = bất thường), KM dùng distances
res_gmm = evaluate("GMM", y_true, y_pred_gmm, -log_probs)
res_km  = evaluate("K-Means", y_true, y_pred_km, distances)

# ── Biểu đồ 5: So sánh toàn diện ──────────────────────────────
fig = plt.figure(figsize=(20, 14))
fig.suptitle("BƯỚC 5 - SO SÁNH TOÀN DIỆN: GMM vs K-Means",
             fontsize=14, fontweight='bold')
gs5 = gridspec.GridSpec(3, 4, figure=fig, hspace=0.5, wspace=0.4)

# ── ROC Curves
ax = fig.add_subplot(gs5[0, :2])
for res, color, ls in [(res_gmm, '#9C27B0', '-'), (res_km, '#FF5722', '--')]:
    fpr, tpr, _ = roc_curve(y_true, res['scores'])
    ax.plot(fpr, tpr, color=color, lw=2, linestyle=ls,
            label=f"{res['name']} (AUC={res['auc']:.4f})")
ax.plot([0,1],[0,1],'k:', lw=1, label='Random')
ax.fill_between(*roc_curve(y_true, res_gmm['scores'])[:2], alpha=0.1, color='#9C27B0')
ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate", fontsize=11)
ax.set_title("Đường cong ROC\n(AUC càng cao càng tốt)", fontsize=12)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# ── Precision-Recall Curves
ax = fig.add_subplot(gs5[0, 2:])
for res, color, ls in [(res_gmm, '#9C27B0', '-'), (res_km, '#FF5722', '--')]:
    prec, rec, _ = precision_recall_curve(y_true, res['scores'])
    ax.plot(rec, prec, color=color, lw=2, linestyle=ls,
            label=f"{res['name']} (AP={res['ap']:.4f})")
ax.set_xlabel("Recall (Độ nhạy)", fontsize=11)
ax.set_ylabel("Precision (Độ chính xác)", fontsize=11)
ax.set_title("Đường cong Precision-Recall\n(AP càng cao càng tốt)", fontsize=12)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# ── Confusion Matrices
for i, (res, title) in enumerate([(res_gmm, "Ma trận nhầm lẫn\nGMM"),
                                   (res_km,  "Ma trận nhầm lẫn\nK-Means")]):
    ax = fig.add_subplot(gs5[1, i*2:(i+1)*2])
    disp = ConfusionMatrixDisplay(res['cm'],
           display_labels=['Bình thường', 'Gian lận'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(title, fontsize=12)

# ── Bar chart so sánh metrics
ax = fig.add_subplot(gs5[2, :2])
metrics = ['ROC-AUC', 'Avg Precision']
vals_gmm = [res_gmm['auc'], res_gmm['ap']]
vals_km  = [res_km['auc'],  res_km['ap']]
x = np.arange(len(metrics))
w = 0.35
bars1 = ax.bar(x - w/2, vals_gmm, w, label='GMM', color='#9C27B0', alpha=0.85)
bars2 = ax.bar(x + w/2, vals_km,  w, label='K-Means', color='#FF5722', alpha=0.85)
for bar in bars1 + bars2:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
            f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylabel("Score", fontsize=11)
ax.set_ylim(0, 1.1)
ax.set_title("So sánh chỉ số đánh giá\nGMM vs K-Means", fontsize=12)
ax.legend(fontsize=10)
ax.grid(alpha=0.3, axis='y')

# ── Comparison table
ax = fig.add_subplot(gs5[2, 2:])
ax.axis('off')

tn_g, fp_g, fn_g, tp_g = res_gmm['cm'].ravel()
tn_k, fp_k, fn_k, tp_k = res_km['cm'].ravel()

table_data = [
    ['Tiêu chí', 'GMM', 'K-Means', 'Tốt hơn'],
    ['ROC-AUC', f'{res_gmm["auc"]:.4f}', f'{res_km["auc"]:.4f}',
     'GMM' if res_gmm['auc'] > res_km['auc'] else 'K-Means'],
    ['Avg Precision', f'{res_gmm["ap"]:.4f}', f'{res_km["ap"]:.4f}',
     'GMM' if res_gmm['ap'] > res_km['ap'] else 'K-Means'],
    ['True Positives', str(tp_g), str(tp_k),
     'GMM' if tp_g > tp_k else 'K-Means'],
    ['False Positives', str(fp_g), str(fp_k),
     'K-Means' if fp_k < fp_g else 'GMM'],
    ['Phương pháp', 'Xác suất', 'Khoảng cách', '-'],
    ['Loại mô hình', 'Soft/Prob', 'Hard/Dist', '-'],
    ['Phù hợp với', 'Dữ liệu phức\ntạp, nhiều cụm', 'Dữ liệu\nđơn giản', '-'],
]

tbl = ax.table(cellText=table_data[1:], colLabels=table_data[0],
               cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor('#37474F')
        cell.set_text_props(color='white', fontweight='bold')
    elif c == 3 and r > 0:
        v = table_data[r+1][3]
        cell.set_facecolor('#A5D6A7' if v == 'GMM' else
                           '#FFCC02' if v == 'K-Means' else '#ECEFF1')
    elif r % 2 == 0:
        cell.set_facecolor('#F5F5F5')
ax.set_title("Bảng so sánh chi tiết\nGMM vs K-Means", fontsize=12, pad=10)

save(fig, "05_comparison.png")

# ─────────────────────────────────────────────────────────────
# BƯỚC 6: PHÂN TÍCH CÁC GIAO DỊCH BỊ GẮN CỜ
# ─────────────────────────────────────────────────────────────
print("\n[BƯỚC 6] Phân tích chi tiết các giao dịch bị GMM gắn cờ...")

df_eval = df_sample.copy()
df_eval['log_prob']  = log_probs
df_eval['gmm_flag']  = y_pred_gmm
df_eval['km_flag']   = y_pred_km
df_eval['km_dist']   = distances

# Top 20 giao dịch nghi ngờ nhất theo GMM
top_suspicious = df_eval.nsmallest(20, 'log_prob')[
    ['Time','Amount','log_prob','Class','gmm_flag','km_flag']].reset_index(drop=True)
print("\n  Top 20 giao dịch nghi ngờ nhất (GMM):")
print(top_suspicious.to_string())

# ── Biểu đồ 6: Phân tích giao dịch được gắn cờ ────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("BƯỚC 6 - PHÂN TÍCH CHI TIẾT: Giao dịch bị GMM gắn cờ",
             fontsize=13, fontweight='bold')

# Phân phối log_prob theo nhóm
ax = axes[0, 0]
grps = {
    'Bình thường (đúng)': df_eval[(df_eval['Class']==0)&(df_eval['gmm_flag']==0)]['log_prob'],
    'Cảnh báo nhầm (FP)': df_eval[(df_eval['Class']==0)&(df_eval['gmm_flag']==1)]['log_prob'],
    'Phát hiện đúng (TP)': df_eval[(df_eval['Class']==1)&(df_eval['gmm_flag']==1)]['log_prob'],
    'Bỏ sót (FN)':         df_eval[(df_eval['Class']==1)&(df_eval['gmm_flag']==0)]['log_prob'],
}
colors_grp = ['#2196F3', '#FF9800', '#4CAF50', '#F44336']
for (lbl, data), col in zip(grps.items(), colors_grp):
    if len(data):
        ax.hist(data.clip(-300, 50), bins=60, density=True, alpha=0.65,
                color=col, label=f'{lbl} (n={len(data):,})')
ax.axvline(threshold, color='black', linestyle='--', linewidth=2, label='Ngưỡng')
ax.set_xlabel("Log-Likelihood", fontsize=10)
ax.set_title("Phân phối log-likelihood\ntheo nhóm dự đoán", fontsize=11)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Scatter: Amount vs log_prob
ax = axes[0, 1]
for flag, col, lbl in [(0,'#2196F3','Bình thường'),(1,'#F44336','Gian lận thật')]:
    mask = y_true == flag
    ax.scatter(df_eval.loc[mask, 'log_prob'].clip(-300, 50),
               df_eval.loc[mask, 'Amount'],
               c=col, s=5 if flag==0 else 40, alpha=0.4 if flag==0 else 0.8,
               label=lbl)
ax.axvline(threshold, color='black', linestyle='--', linewidth=1.5, label='Ngưỡng')
ax.set_xlabel("Log-Likelihood", fontsize=10)
ax.set_ylabel("Số tiền GD (Amount)", fontsize=10)
ax.set_title("Mối quan hệ Amount vs Log-Likelihood", fontsize=11)
ax.legend(fontsize=9, markerscale=3)
ax.grid(alpha=0.3)

# GMM vs KMeans venn-like scatter
ax = axes[0, 2]
cat = np.where(
    (y_pred_gmm==1) & (y_pred_km==1), 'Cả 2 đều gắn cờ',
    np.where((y_pred_gmm==1) & (y_pred_km==0), 'Chỉ GMM',
    np.where((y_pred_gmm==0) & (y_pred_km==1), 'Chỉ K-Means', 'Không gắn cờ'))
)
cat_colors = {'Cả 2 đều gắn cờ': '#F44336',
              'Chỉ GMM': '#9C27B0',
              'Chỉ K-Means': '#FF9800',
              'Không gắn cờ': '#B0BEC5'}
for lbl, col in cat_colors.items():
    mask = cat == lbl
    s = 50 if 'gắn cờ' in lbl else 5
    al= 0.8 if 'gắn cờ' in lbl else 0.15
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=col, s=s, alpha=al, label=lbl)
ax.set_title("So sánh cờ GMM vs K-Means\n(PCA 2D)", fontsize=11)
ax.set_xlabel(f"PC1", fontsize=10)
ax.set_ylabel(f"PC2", fontsize=10)
ax.legend(fontsize=9, markerscale=2)
ax.grid(alpha=0.3)

# Radar chart metrics (per-class)
from matplotlib.patches import FancyArrowPatch

def plot_radar(ax, values_list, labels_models, cat_labels, title):
    N = len(cat_labels)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cat_labels, fontsize=9)
    ax.set_ylim(0, 1)
    colors_r = ['#9C27B0', '#FF5722']
    for vals, lbl, col in zip(values_list, labels_models, colors_r):
        v = vals + vals[:1]
        ax.plot(angles, v, 'o-', linewidth=2, color=col, label=lbl)
        ax.fill(angles, v, alpha=0.15, color=col)
    ax.set_title(title, fontsize=11, pad=15)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=9)

from sklearn.metrics import recall_score, precision_score, f1_score
def safe_metric(fn, *args, **kw):
    try: return fn(*args, **kw)
    except: return 0.0

cat_labels_r = ['Recall\n(fraud)', 'Precision\n(fraud)', 'F1\n(fraud)',
                 'ROC-AUC', 'Avg Prec']
vals_gmm_r = [safe_metric(recall_score, y_true, y_pred_gmm),
              safe_metric(precision_score, y_true, y_pred_gmm),
              safe_metric(f1_score, y_true, y_pred_gmm),
              res_gmm['auc'], res_gmm['ap']]
vals_km_r  = [safe_metric(recall_score, y_true, y_pred_km),
              safe_metric(precision_score, y_true, y_pred_km),
              safe_metric(f1_score, y_true, y_pred_km),
              res_km['auc'], res_km['ap']]

ax_r = fig.add_subplot(2, 3, 4, projection='polar')  # uses axes[1,0] position
plot_radar(ax_r, [vals_gmm_r, vals_km_r],
           ['GMM', 'K-Means'], cat_labels_r,
           "Radar Chart: GMM vs K-Means\n(5 chỉ số)")

# Amount của TP vs FN (GMM)
ax = axes[1, 1]
tp_amounts = df_eval[(df_eval['Class']==1)&(df_eval['gmm_flag']==1)]['Amount']
fn_amounts = df_eval[(df_eval['Class']==1)&(df_eval['gmm_flag']==0)]['Amount']
ax.hist(tp_amounts, bins=30, alpha=0.7, color='#4CAF50',
        label=f'Phát hiện đúng TP (n={len(tp_amounts)})', density=True)
ax.hist(fn_amounts, bins=30, alpha=0.7, color='#F44336',
        label=f'Bỏ sót FN (n={len(fn_amounts)})', density=True)
ax.set_xlabel("Số tiền giao dịch (USD)", fontsize=10)
ax.set_ylabel("Mật độ", fontsize=10)
ax.set_title("Amount của Gian lận: TP vs FN\n(GMM)", fontsize=11)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Timeline: giao dịch gian lận bị phát hiện
ax = axes[1, 2]
fraud_df = df_eval[df_eval['Class']==1].copy()
fraud_df['hour'] = fraud_df['Time'] / 3600
ax.scatter(fraud_df[fraud_df['gmm_flag']==1]['hour'],
           fraud_df[fraud_df['gmm_flag']==1]['Amount'],
           c='#4CAF50', s=60, marker='o', label='Phát hiện (TP)', zorder=5, alpha=0.8)
ax.scatter(fraud_df[fraud_df['gmm_flag']==0]['hour'],
           fraud_df[fraud_df['gmm_flag']==0]['Amount'],
           c='#F44336', s=60, marker='x', linewidths=2, label='Bỏ sót (FN)', zorder=5)
ax.set_xlabel("Thời gian (giờ)", fontsize=10)
ax.set_ylabel("Số tiền (USD)", fontsize=10)
ax.set_title("Timeline Giao dịch Gian lận\nGMM: TP (●) vs FN (×)", fontsize=11)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

save(fig, "06_detailed_analysis.png")

# ─────────────────────────────────────────────────────────────
# BƯỚC 7: TÓM TẮT & KẾT LUẬN
# ─────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  KẾT QUẢ TỔNG KẾT")
print("="*65)

for res in [res_gmm, res_km]:
    tn, fp, fn, tp = res['cm'].ravel()
    recall    = tp / (tp + fn) if (tp+fn)>0 else 0
    precision = tp / (tp + fp) if (tp+fp)>0 else 0
    f1 = 2*recall*precision/(recall+precision) if (recall+precision)>0 else 0
    print(f"\n  {res['name']}:")
    print(f"    ROC-AUC        : {res['auc']:.4f}")
    print(f"    Avg Precision  : {res['ap']:.4f}")
    print(f"    Recall (fraud) : {recall:.4f}  ({tp}/{tp+fn} gian lận phát hiện)")
    print(f"    Precision(fraud): {precision:.4f}")
    print(f"    F1-Score       : {f1:.4f}")
    print(f"    False Alarm FP : {fp:,}")

print("\n  NHẬN XÉT:")
print("  ─ GMM mô hình hoá hình dạng xác suất của dữ liệu → phát hiện")
print("    điểm bất thường dựa trên mật độ xác suất thấp.")
print("  ─ K-Means chỉ dựa vào khoảng cách Euclidean → đơn giản hơn nhưng")
print("    không nắm bắt được hình dạng phức tạp của dữ liệu.")
print("  ─ Với bộ dữ liệu có nhiều chiều và phân phối phi tuyến như gian lận")
print("    thẻ tín dụng, GMM thường cho kết quả tốt hơn K-Means.")
print(f"\n  Các biểu đồ đã lưu tại: {OUT_DIR}")
print("="*65)
print("\n  DANH SÁCH BIỂU ĐỒ:")
plot_files = [
    "01_dataset_overview.png   → Tổng quan bộ dữ liệu, phân phối nhãn, Amount, Time",
    "02_gmm_training.png       → BIC để chọn K, phân phối log-likelihood, PCA scatter",
    "03_gmm_components_2d.png  → Visualize các Gaussian components trong PCA 2D",
    "04_kmeans_analysis.png    → K-Means clustering, phân phối khoảng cách, gắn cờ",
    "05_comparison.png         → ROC, Precision-Recall, Confusion Matrix, bảng so sánh",
    "06_detailed_analysis.png  → Phân tích sâu: timeline, TP vs FN, overlap GMM-KMeans",
]
for f in plot_files:
    print(f"    • {f}")
