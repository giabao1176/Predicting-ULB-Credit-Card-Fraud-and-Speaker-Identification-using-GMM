#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
=============================================================================
 NHẬN DẠNG NGƯỜI NÓI BẰNG GMM (Gaussian Mixture Model)
 Sử dụng bộ dữ liệu Free Spoken Digit Dataset (FSDD)
=============================================================================

Mô tả:
- GMM từng là "tiêu chuẩn vàng" trong xử lý âm thanh trước Deep Learning
- Giọng nói mỗi người có đặc trưng tần số (MFCCs) phân bố riêng biệt
- Ta dùng GMM để học "dấu vân tay âm thanh" cho từng người
- Ứng dụng: Xác thực danh tính qua giọng nói (ngân hàng, bảo mật)

Dataset: FSDD - 6 speakers, 3000 recordings, 8kHz WAV
"""

# ============================================================================
# CELL 1: Import thư viện
# ============================================================================

import os
import glob
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Audio processing
import librosa
import librosa.display

# Machine Learning
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Visualization
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

print("✅ Đã import tất cả thư viện thành công!")
print(f"   - librosa version: {librosa.__version__}")
print(f"   - numpy version: {np.__version__}")

# ============================================================================
# CELL 2: Cấu hình và tải dữ liệu
# ============================================================================

# Đường dẫn đến thư mục recordings
RECORDINGS_DIR = './archive/recordings'

# Các speaker trong dataset
SPEAKERS = ['george', 'jackson', 'lucas', 'nicolas', 'theo', 'yweweler']

# Thông tin metadata về speakers
SPEAKER_INFO = {
    'george':   {'gender': 'male', 'accent': 'GRC/Greek'},
    'jackson':  {'gender': 'male', 'accent': 'USA/Neutral'},
    'lucas':    {'gender': 'male', 'accent': 'DEU/German'},
    'nicolas':  {'gender': 'male', 'accent': 'BEL/French'},
    'theo':     {'gender': 'male', 'accent': 'USA/Neutral'},
    'yweweler': {'gender': 'male', 'accent': 'DEU/German'},
}

# Màu sắc cho từng speaker
SPEAKER_COLORS = {
    'george':   '#e74c3c',
    'jackson':  '#3498db',
    'lucas':    '#2ecc71',
    'nicolas':  '#f39c12',
    'theo':     '#9b59b6',
    'yweweler': '#1abc9c',
}

# Đếm số file cho mỗi speaker
print("\n📊 THỐNG KÊ BỘ DỮ LIỆU FSDD")
print("=" * 50)
total_files = 0
for speaker in SPEAKERS:
    files = glob.glob(os.path.join(RECORDINGS_DIR, f'*_{speaker}_*.wav'))
    count = len(files)
    total_files += count
    accent = SPEAKER_INFO[speaker]['accent']
    print(f"   🎤 {speaker:10s} | {count:4d} files | Accent: {accent}")
print("=" * 50)
print(f"   📁 Tổng cộng: {total_files} files")

# ============================================================================
# CELL 3: Trích xuất đặc trưng MFCC
# ============================================================================

def extract_mfcc(file_path, n_mfcc=13, sr=8000):
    """
    Trích xuất MFCC (Mel-Frequency Cepstral Coefficients) từ file .wav
    
    MFCC là gì?
    - Là đặc trưng phổ biến nhất trong xử lý âm thanh/giọng nói
    - Mô phỏng cách tai người cảm nhận âm thanh
    - Mỗi hệ số MFCC đại diện cho một "băng tần" của giọng nói
    
    Parameters:
    -----------
    file_path : str - đường dẫn file WAV
    n_mfcc : int - số hệ số MFCC (thường 13-20)
    sr : int - sampling rate (8000 Hz cho FSDD)
    
    Returns:
    --------
    mfccs : numpy array - ma trận MFCC (n_mfcc x T) với T là số frame
    """
    # Đọc file âm thanh
    y, sr = librosa.load(file_path, sr=sr)
    
    # Trích xuất MFCC
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    # Kết hợp: MFCC + Delta + Delta-Delta = 39 chiều
    # Xử lý lỗi width > số lượng frame cho các file quá ngắn
    n_frames = mfccs.shape[1]
    
    # Mặc định width của delta là 9, yêu cầu width <= n_frames và là số lẻ >= 3
    delta_width = min(9, n_frames if n_frames % 2 != 0 else n_frames - 1)
    
    if delta_width >= 3:
        delta_mfccs = librosa.feature.delta(mfccs, width=delta_width)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2, width=delta_width)
        combined = np.vstack([mfccs, delta_mfccs, delta2_mfccs])
    else:
        # Nếu audio quá ngắn (chưa tới 3 frames), chỉ dùng MFCC và pad bằng 0 cho delta/delta2
        delta_pad = np.zeros_like(mfccs)
        combined = np.vstack([mfccs, delta_pad, delta_pad])
    
    return combined.T  # Transpose: (T, 39) - mỗi hàng là 1 frame

print("\n🔬 TRÍCH XUẤT ĐẶC TRƯNG MFCC")
print("=" * 50)

# Trích xuất MFCC cho tất cả speaker
speaker_features = {}  # {speaker_name: [list of MFCC matrices]}

for speaker in SPEAKERS:
    files = sorted(glob.glob(os.path.join(RECORDINGS_DIR, f'*_{speaker}_*.wav')))
    features_list = []
    
    for f in files:
        mfcc = extract_mfcc(f)
        features_list.append(mfcc)
    
    speaker_features[speaker] = features_list
    n_frames = sum(m.shape[0] for m in features_list)
    print(f"   🎤 {speaker:10s} | {len(features_list):4d} files | {n_frames:6d} frames | {features_list[0].shape[1]} features/frame")

print(f"\n   ✅ Mỗi frame có {features_list[0].shape[1]} đặc trưng (13 MFCC + 13 Delta + 13 Delta²)")

# ============================================================================
# CELL 4: Biểu đồ 1 - Trực quan hóa dạng sóng & MFCC
# ============================================================================

print("\n🎨 BIỂU ĐỒ 1: DẠNG SÓNG & MFCC CỦA 6 SPEAKER")
print("=" * 50)

fig, axes = plt.subplots(6, 2, figsize=(16, 20))
fig.suptitle('BIỂU ĐỒ 1: Dạng sóng & Đặc trưng MFCC của 6 Speaker\n(Cùng nói số "5")', 
             fontsize=16, fontweight='bold', y=1.01)

for i, speaker in enumerate(SPEAKERS):
    # Tìm file nói số "5" của speaker
    sample_file = os.path.join(RECORDINGS_DIR, f'5_{speaker}_10.wav')
    y, sr = librosa.load(sample_file, sr=8000)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    color = SPEAKER_COLORS[speaker]
    accent = SPEAKER_INFO[speaker]['accent']
    
    # Dạng sóng
    ax1 = axes[i, 0]
    librosa.display.waveshow(y, sr=sr, ax=ax1, color=color, alpha=0.8)
    ax1.set_title(f'{speaker.upper()} ({accent}) - Waveform', fontweight='bold', color=color)
    ax1.set_ylabel('Amplitude')
    if i < 5:
        ax1.set_xlabel('')
    
    # MFCC
    ax2 = axes[i, 1]
    img = librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=ax2, cmap='RdBu_r')
    ax2.set_title(f'{speaker.upper()} - MFCC Spectrogram', fontweight='bold', color=color)
    ax2.set_ylabel('MFCC Coeff')
    if i < 5:
        ax2.set_xlabel('')
    plt.colorbar(img, ax=ax2, format='%+2.0f')

plt.tight_layout()
plt.savefig('plots/speaker_waveform_mfcc.png', dpi=150, bbox_inches='tight')
plt.show()
print("   ✅ Đã lưu biểu đồ: plots/speaker_waveform_mfcc.png")

# ============================================================================
# CELL 5: Chia tập Train/Test
# ============================================================================

print("\n📂 CHIA TẬP TRAIN / TEST")
print("=" * 50)

# Theo quy ước FSDD: index 0-4 = test, index 5-49 = train
train_features = {}  # {speaker: concatenated MFCC matrix}
test_features = {}   # {speaker: list of MFCC matrices per file}
test_labels = []
test_file_features = []

for speaker in SPEAKERS:
    files = sorted(glob.glob(os.path.join(RECORDINGS_DIR, f'*_{speaker}_*.wav')))
    
    train_mfcc_list = []
    test_mfcc_list = []
    
    for f in files:
        # Lấy index từ tên file: {digit}_{speaker}_{index}.wav
        basename = os.path.basename(f)
        parts = basename.replace('.wav', '').split('_')
        idx = int(parts[2])
        
        mfcc = extract_mfcc(f)
        
        if idx < 5:  # Test set
            test_mfcc_list.append(mfcc)
            test_labels.append(speaker)
            test_file_features.append(mfcc)
        else:  # Train set
            train_mfcc_list.append(mfcc)
    
    # Gộp tất cả frames của train thành 1 ma trận lớn cho mỗi speaker
    train_features[speaker] = np.vstack(train_mfcc_list)
    test_features[speaker] = test_mfcc_list
    
    print(f"   🎤 {speaker:10s} | Train: {train_features[speaker].shape[0]:6d} frames ({len(train_mfcc_list):3d} files)")
    print(f"   {'':10s}   | Test:  {sum(m.shape[0] for m in test_mfcc_list):6d} frames ({len(test_mfcc_list):3d} files)")

print(f"\n   📊 Tổng test samples: {len(test_labels)} files")

# ============================================================================
# CELL 6: Huấn luyện GMM cho từng Speaker
# ============================================================================

print("\n🧠 HUẤN LUYỆN GMM CHO TỪNG SPEAKER")
print("=" * 50)
print("""
📌 Cách hoạt động:
   - Mỗi speaker được mô hình hóa bằng 1 GMM riêng
   - GMM học phân phối MFCC của speaker đó
   - Khi nhận dạng: tính xác suất audio khớp với từng GMM
   - Speaker nào cho xác suất cao nhất → đó là người nói
""")

# Chọn số thành phần GMM tối ưu bằng BIC
print("📐 Tìm số thành phần GMM tối ưu bằng BIC...")

# Thử nghiệm với 1 speaker để chọn K tối ưu
test_speaker = 'george'
bic_scores = []
n_components_range = range(2, 17, 2)

for n in n_components_range:
    gmm = GaussianMixture(n_components=n, covariance_type='diag', 
                           max_iter=200, random_state=42)
    gmm.fit(train_features[test_speaker])
    bic_scores.append(gmm.bic(train_features[test_speaker]))
    print(f"   K={n:2d} | BIC = {bic_scores[-1]:.2f}")

optimal_k = list(n_components_range)[np.argmin(bic_scores)]
print(f"\n   ✅ Số thành phần tối ưu (BIC thấp nhất): K = {optimal_k}")

# Huấn luyện GMM cho từng speaker
N_COMPONENTS = optimal_k
speaker_gmms = {}

print(f"\n🔧 Huấn luyện GMM (K={N_COMPONENTS}) cho tất cả speakers...")
for speaker in SPEAKERS:
    gmm = GaussianMixture(
        n_components=N_COMPONENTS,
        covariance_type='diag',
        max_iter=200,
        n_init=3,
        random_state=42
    )
    gmm.fit(train_features[speaker])
    speaker_gmms[speaker] = gmm
    
    avg_ll = gmm.score(train_features[speaker])
    print(f"   🎤 {speaker:10s} | Avg Log-Likelihood (train): {avg_ll:.4f}")

print("\n   ✅ Đã huấn luyện xong GMM cho tất cả 6 speakers!")

# ============================================================================
# CELL 7: Biểu đồ 2 - BIC Score & Log-Likelihood
# ============================================================================

print("\n🎨 BIỂU ĐỒ 2: BIC SCORE & LOG-LIKELIHOOD")
print("=" * 50)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('BIỂU ĐỒ 2: Lựa chọn số thành phần GMM & Log-Likelihood trên tập Train', 
             fontsize=14, fontweight='bold')

# BIC Score
ax1 = axes[0]
ax1.plot(list(n_components_range), bic_scores, 'bo-', linewidth=2, markersize=8, label='BIC Score')
ax1.axvline(x=optimal_k, color='r', linestyle='--', linewidth=2, label=f'Optimal K={optimal_k}')
ax1.fill_between(list(n_components_range), min(bic_scores), bic_scores, alpha=0.1, color='blue')
ax1.set_xlabel('Số thành phần GMM (K)')
ax1.set_ylabel('BIC Score')
ax1.set_title('BIC Score vs. Số thành phần\n(Thấp hơn = Tốt hơn)', fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Log-Likelihood trên train cho từng speaker
ax2 = axes[1]
speakers_list = []
ll_values = []
colors_list = []

for speaker in SPEAKERS:
    ll = speaker_gmms[speaker].score(train_features[speaker])
    speakers_list.append(speaker.upper())
    ll_values.append(ll)
    colors_list.append(SPEAKER_COLORS[speaker])

bars = ax2.bar(speakers_list, ll_values, color=colors_list, edgecolor='white', linewidth=2)
ax2.set_ylabel('Avg Log-Likelihood')
ax2.set_title(f'Log-Likelihood trên tập Train (K={N_COMPONENTS})', fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# Thêm giá trị lên thanh
for bar, val in zip(bars, ll_values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('plots/speaker_bic_loglik.png', dpi=150, bbox_inches='tight')
plt.show()
print("   ✅ Đã lưu biểu đồ: plots/speaker_bic_loglik.png")

# ============================================================================
# CELL 8: Nhận dạng người nói (Speaker Identification)
# ============================================================================

print("\n🎯 NHẬN DẠNG NGƯỜI NÓI (SPEAKER IDENTIFICATION)")
print("=" * 50)
print("""
📌 Quy trình nhận dạng:
   1. Trích xuất MFCC từ audio test
   2. Tính log-likelihood với từng GMM speaker
   3. Speaker nào cho log-likelihood cao nhất = người nói
""")

predictions = []
true_labels = []
all_scores = []  # Ma trận score cho visualization

for i, (label, features) in enumerate(zip(test_labels, test_file_features)):
    scores = {}
    for speaker in SPEAKERS:
        # Tính trung bình log-likelihood trên tất cả frames
        score = speaker_gmms[speaker].score(features)
        scores[speaker] = score
    
    # Dự đoán: speaker có score cao nhất
    predicted = max(scores, key=scores.get)
    predictions.append(predicted)
    true_labels.append(label)
    all_scores.append([scores[s] for s in SPEAKERS])

# Tính accuracy
accuracy = accuracy_score(true_labels, predictions)
print(f"\n   🎯 Accuracy trên tập test: {accuracy * 100:.2f}%")
print(f"   📊 Đúng: {sum(p == t for p, t in zip(predictions, true_labels))}/{len(true_labels)} files")

# Classification Report
print("\n📋 CLASSIFICATION REPORT:")
print("-" * 60)
print(classification_report(true_labels, predictions, target_names=[s.upper() for s in SPEAKERS]))

# ============================================================================
# CELL 9: Biểu đồ 3 - Confusion Matrix
# ============================================================================

print("\n🎨 BIỂU ĐỒ 3: CONFUSION MATRIX")
print("=" * 50)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('BIỂU ĐỒ 3: Ma trận nhầm lẫn - Speaker Identification', 
             fontsize=14, fontweight='bold')

# Confusion matrix (counts)
cm = confusion_matrix(true_labels, predictions, labels=SPEAKERS)
ax1 = axes[0]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=[s.upper() for s in SPEAKERS],
            yticklabels=[s.upper() for s in SPEAKERS],
            ax=ax1, linewidths=1, linecolor='white',
            cbar_kws={'label': 'Số lượng'})
ax1.set_xlabel('Dự đoán (Predicted)', fontweight='bold')
ax1.set_ylabel('Thực tế (Actual)', fontweight='bold')
ax1.set_title(f'Confusion Matrix (Counts)\nAccuracy: {accuracy*100:.1f}%', fontweight='bold')

# Confusion matrix (percentages)
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
ax2 = axes[1]
sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='RdYlGn',
            xticklabels=[s.upper() for s in SPEAKERS],
            yticklabels=[s.upper() for s in SPEAKERS],
            ax=ax2, linewidths=1, linecolor='white',
            cbar_kws={'label': '%'}, vmin=0, vmax=100)
ax2.set_xlabel('Dự đoán (Predicted)', fontweight='bold')
ax2.set_ylabel('Thực tế (Actual)', fontweight='bold')
ax2.set_title('Confusion Matrix (Percentages)\nĐường chéo = Nhận đúng', fontweight='bold')

plt.tight_layout()
plt.savefig('plots/speaker_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("   ✅ Đã lưu biểu đồ: plots/speaker_confusion_matrix.png")

# ============================================================================
# CELL 10: Biểu đồ 4 - Log-Likelihood Heatmap (Speaker Verification)
# ============================================================================

print("\n🎨 BIỂU ĐỒ 4: SPEAKER VERIFICATION - LOG-LIKELIHOOD HEATMAP")
print("=" * 50)
print("""
📌 Speaker Verification (Xác thực người nói):
   - Khác với Identification (nhận dạng ai đang nói)
   - Verification: Kiểm tra "Bạn có phải là người X không?"
   - Nếu log-likelihood > ngưỡng → Xác thực thành công ✅
   - Nếu log-likelihood < ngưỡng → Từ chối ❌
""")

# Tính trung bình log-likelihood cho mỗi cặp (speaker_true, speaker_model)
avg_scores = np.zeros((len(SPEAKERS), len(SPEAKERS)))

for i, true_speaker in enumerate(SPEAKERS):
    # Lấy tất cả test files của true_speaker
    indices = [j for j, l in enumerate(true_labels) if l == true_speaker]
    
    for k, model_speaker in enumerate(SPEAKERS):
        scores_list = []
        for idx in indices:
            score = speaker_gmms[model_speaker].score(test_file_features[idx])
            scores_list.append(score)
        avg_scores[i, k] = np.mean(scores_list)

fig, ax = plt.subplots(figsize=(10, 8))
fig.suptitle('BIỂU ĐỒ 4: Trung bình Log-Likelihood cho mỗi cặp Speaker\n(Đường chéo cao = GMM phân biệt tốt)', 
             fontsize=14, fontweight='bold')

sns.heatmap(avg_scores, annot=True, fmt='.2f', cmap='YlOrRd',
            xticklabels=[f'GMM\n{s.upper()}' for s in SPEAKERS],
            yticklabels=[f'Audio\n{s.upper()}' for s in SPEAKERS],
            ax=ax, linewidths=2, linecolor='white',
            cbar_kws={'label': 'Avg Log-Likelihood'})
ax.set_xlabel('Mô hình GMM', fontweight='bold', fontsize=13)
ax.set_ylabel('Audio thực tế', fontweight='bold', fontsize=13)

# Highlight đường chéo
for i in range(len(SPEAKERS)):
    ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='blue', linewidth=3))

plt.tight_layout()
plt.savefig('plots/speaker_verification_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("   ✅ Đã lưu biểu đồ: plots/speaker_verification_heatmap.png")

# ============================================================================
# CELL 11: Biểu đồ 5 - Phân bố MFCC theo Speaker (PCA 2D)
# ============================================================================

print("\n🎨 BIỂU ĐỒ 5: PHÂN BỐ MFCC TRONG KHÔNG GIAN PCA 2D")
print("=" * 50)

from sklearn.decomposition import PCA

# Lấy sample frames từ mỗi speaker (giới hạn 500 frames/speaker)
pca_data = []
pca_labels = []
SAMPLE_SIZE = 500

for speaker in SPEAKERS:
    data = train_features[speaker]
    if len(data) > SAMPLE_SIZE:
        indices = np.random.choice(len(data), SAMPLE_SIZE, replace=False)
        data = data[indices]
    pca_data.append(data)
    pca_labels.extend([speaker] * len(data))

pca_data = np.vstack(pca_data)

# PCA giảm chiều
scaler = StandardScaler()
pca_scaled = scaler.fit_transform(pca_data)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(pca_scaled)

fig, ax = plt.subplots(figsize=(12, 10))
fig.suptitle('BIỂU ĐỒ 5: Phân bố đặc trưng MFCC trong không gian PCA 2D\n(Mỗi điểm = 1 frame âm thanh)', 
             fontsize=14, fontweight='bold')

for speaker in SPEAKERS:
    mask = np.array(pca_labels) == speaker
    ax.scatter(pca_result[mask, 0], pca_result[mask, 1],
              c=SPEAKER_COLORS[speaker], label=speaker.upper(),
              alpha=0.4, s=15, edgecolors='none')

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontweight='bold')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontweight='bold')
ax.legend(fontsize=12, markerscale=3, loc='upper right',
         title='Speaker', title_fontsize=13)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/speaker_pca_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("   ✅ Đã lưu biểu đồ: plots/speaker_pca_distribution.png")

# ============================================================================
# CELL 12: Biểu đồ 6 - Demo Speaker Verification (Mô phỏng ngân hàng)
# ============================================================================

print("\n🎨 BIỂU ĐỒ 6: MÔ PHỎNG HỆ THỐNG XÁC THỰC GIỌNG NÓI NGÂN HÀNG")
print("=" * 50)
print("""
📌 Kịch bản mô phỏng:
   - Hệ thống ngân hàng đã đăng ký giọng nói "Jackson"
   - 6 người khác nhau gọi đến và nói "Tôi là Jackson"
   - Hệ thống dùng GMM để xác thực: Chấp nhận hay Từ chối?
""")

# Mô phỏng verification cho "Jackson"
target_speaker = 'jackson'
target_gmm = speaker_gmms[target_speaker]

# Tính log-likelihood threshold dựa trên training data
train_scores = target_gmm.score_samples(train_features[target_speaker])
threshold = np.percentile(train_scores, 5)  # Ngưỡng: percentile 5%

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle(f'BIỂU ĐỒ 6: Mô phỏng xác thực giọng nói - Tài khoản "{target_speaker.upper()}"',
             fontsize=16, fontweight='bold')

for idx, speaker in enumerate(SPEAKERS):
    ax = axes[idx // 3, idx % 3]
    
    # Lấy 1 file test
    test_files = sorted(glob.glob(os.path.join(RECORDINGS_DIR, f'5_{speaker}_0.wav')))
    if test_files:
        mfcc = extract_mfcc(test_files[0])
        scores = target_gmm.score_samples(mfcc)
        avg_score = np.mean(scores)
        
        # Vẽ histogram log-likelihood
        ax.hist(scores, bins=30, color=SPEAKER_COLORS[speaker], alpha=0.7, 
                edgecolor='white', density=True)
        ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'Ngưỡng: {threshold:.1f}')
        ax.axvline(x=avg_score, color='black', linestyle='-', linewidth=2, label=f'Score: {avg_score:.1f}')
        
        # Kết quả
        is_verified = avg_score > threshold
        status = "✅ CHẤP NHẬN" if (speaker == target_speaker) and is_verified else \
                 "✅ CHẤP NHẬN" if is_verified else "❌ TỪ CHỐI"
        status_color = 'green' if '✅' in status else 'red'
        
        ax.set_title(f'Người nói: {speaker.upper()}\n{status}',
                     fontweight='bold', color=status_color, fontsize=13)
        ax.set_xlabel('Log-Likelihood')
        ax.set_ylabel('Density')
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/speaker_verification_demo.png', dpi=150, bbox_inches='tight')
plt.show()
print("   ✅ Đã lưu biểu đồ: plots/speaker_verification_demo.png")

# ============================================================================
# CELL 13: Biểu đồ 7 - So sánh MFCC trung bình giữa các Speaker
# ============================================================================

print("\n🎨 BIỂU ĐỒ 7: SO SÁNH MFCC TRUNG BÌNH GIỮA CÁC SPEAKER")
print("=" * 50)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('BIỂU ĐỒ 7: Trung bình & Độ lệch chuẩn MFCC - "Dấu vân tay âm thanh" của mỗi Speaker',
             fontsize=14, fontweight='bold')

for idx, speaker in enumerate(SPEAKERS):
    ax = axes[idx // 3, idx % 3]
    
    mean_mfcc = np.mean(train_features[speaker][:, :13], axis=0)
    std_mfcc = np.std(train_features[speaker][:, :13], axis=0)
    
    x = np.arange(13)
    ax.bar(x, mean_mfcc, yerr=std_mfcc, color=SPEAKER_COLORS[speaker],
           edgecolor='white', alpha=0.8, capsize=3)
    ax.set_title(f'{speaker.upper()} ({SPEAKER_INFO[speaker]["accent"]})',
                 fontweight='bold', color=SPEAKER_COLORS[speaker])
    ax.set_xlabel('MFCC Coefficient')
    ax.set_ylabel('Value')
    ax.set_xticks(x)
    ax.set_xticklabels([f'C{i}' for i in range(13)], fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('plots/speaker_mfcc_fingerprint.png', dpi=150, bbox_inches='tight')
plt.show()
print("   ✅ Đã lưu biểu đồ: plots/speaker_mfcc_fingerprint.png")

# ============================================================================
# CELL 14: Biểu đồ 8 - Accuracy theo số thành phần GMM
# ============================================================================

print("\n🎨 BIỂU ĐỒ 8: ẢNH HƯỞNG SỐ THÀNH PHẦN GMM ĐẾN ACCURACY")
print("=" * 50)

k_range = [2, 4, 8, 12, 16, 24, 32]
accuracies = []

for k in k_range:
    # Train GMMs
    temp_gmms = {}
    for speaker in SPEAKERS:
        gmm = GaussianMixture(n_components=k, covariance_type='diag',
                               max_iter=200, random_state=42)
        gmm.fit(train_features[speaker])
        temp_gmms[speaker] = gmm
    
    # Predict
    correct = 0
    for label, features in zip(test_labels, test_file_features):
        scores = {s: temp_gmms[s].score(features) for s in SPEAKERS}
        pred = max(scores, key=scores.get)
        if pred == label:
            correct += 1
    
    acc = correct / len(test_labels) * 100
    accuracies.append(acc)
    print(f"   K={k:2d} | Accuracy: {acc:.2f}%")

fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle('BIỂU ĐỒ 8: Accuracy theo số thành phần GMM',
             fontsize=14, fontweight='bold')

ax.plot(k_range, accuracies, 'bo-', linewidth=2, markersize=10, label='Accuracy')
ax.fill_between(k_range, 0, accuracies, alpha=0.1, color='blue')

# Highlight best
best_idx = np.argmax(accuracies)
ax.plot(k_range[best_idx], accuracies[best_idx], 'r*', markersize=20,
        label=f'Best: K={k_range[best_idx]}, Acc={accuracies[best_idx]:.1f}%')

ax.set_xlabel('Số thành phần GMM (K)', fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontweight='bold')
ax.set_ylim([0, 105])
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_xticks(k_range)

plt.tight_layout()
plt.savefig('plots/speaker_accuracy_vs_k.png', dpi=150, bbox_inches='tight')
plt.show()
print("   ✅ Đã lưu biểu đồ: plots/speaker_accuracy_vs_k.png")

# ============================================================================
# CELL 15: Tổng kết
# ============================================================================

print("\n" + "=" * 60)
print("📊 TỔNG KẾT: NHẬN DẠNG NGƯỜI NÓI BẰNG GMM")
print("=" * 60)

print(f"""
🎯 KẾT QUẢ:
   - Dataset: Free Spoken Digit Dataset (FSDD)
   - Số speaker: 6 (george, jackson, lucas, nicolas, theo, yweweler)
   - Số recordings: 3,000 files WAV (8kHz)
   - Đặc trưng: 39 chiều (13 MFCC + 13 Delta + 13 Delta²)
   - Mô hình: GMM với K={N_COMPONENTS} components, covariance='diag'
   - Accuracy: {accuracy * 100:.2f}%

📌 CÁCH GMM HOẠT ĐỘNG TRONG NHẬN DẠNG GIỌNG NÓI:
   1. ĐĂNG KÝ: Người dùng ghi âm mẫu → Trích xuất MFCC → Train GMM
   2. XÁC THỰC: Ghi âm mới → Trích xuất MFCC → Tính score với GMM
   3. QUYẾT ĐỊNH: Score > ngưỡng → Xác thực thành công ✅

📌 ƯU ĐIỂM CỦA GMM:
   ✅ Không cần nhiều dữ liệu training
   ✅ Tính toán nhanh, dễ triển khai
   ✅ Hoạt động tốt với đặc trưng MFCC
   ✅ Có thể cập nhật mô hình (incremental learning)

📌 HẠN CHẾ:
   ❌ Nhạy cảm với nhiễu môi trường
   ❌ Khó mở rộng với hàng triệu speakers
   ❌ Đã được thay thế bởi i-vectors, x-vectors (Deep Learning)

📊 BIỂU ĐỒ ĐÃ TẠO:
   1. Dạng sóng & MFCC của 6 speakers
   2. BIC Score & Log-Likelihood
   3. Confusion Matrix
   4. Speaker Verification - Log-Likelihood Heatmap
   5. Phân bố MFCC trong PCA 2D
   6. Mô phỏng xác thực giọng nói ngân hàng
   7. Dấu vân tay MFCC trung bình
   8. Accuracy theo số thành phần GMM
""")
print("=" * 60)
print("✅ HO À N T H À N H!")
print("=" * 60)
