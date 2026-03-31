# 🧠 Predicting ULB Credit Card Fraud and Speaker Identification using GMM

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-F7931E?style=for-the-badge&logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?style=for-the-badge&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Ứng dụng Gaussian Mixture Model (GMM) vào hai bài toán thực tế: Phát hiện gian lận thẻ tín dụng và Nhận dạng người nói**

</div>

---

## 📌 Giới thiệu

Dự án này nghiên cứu và triển khai thuật toán **Gaussian Mixture Model (GMM)** — một phương pháp học máy xác suất mạnh mẽ — vào hai bài toán ứng dụng hoàn toàn khác nhau:

| Bài toán | Dữ liệu | Phương pháp |
|----------|---------|-------------|
| 💳 **Credit Card Fraud Detection** | ULB Kaggle Dataset (284,807 giao dịch) | GMM Anomaly Detection vs. K-Means |
| 🎙️ **Speaker Identification** | Free Spoken Digit Dataset – FSDD (3,000 files WAV) | GMM + MFCC Features |

Dự án bao gồm:
- ✅ Phân tích lý thuyết và toán học của GMM
- ✅ Triển khai đầy đủ hai pipeline machine learning
- ✅ So sánh hiệu năng GMM với K-Means
- ✅ Ứng dụng web tương tác minh họa thuật toán bằng **Streamlit**

---

## 📂 Cấu trúc thư mục

```
GMM_Speaker_ID/
│
├── README.md                        # Mô tả dự án
├── requirements.txt                 # Thư viện cần cài
│
├── data/                            # Dữ liệu (xem hướng dẫn tải bên dưới)
│   └── archive/                     # Dữ liệu FSDD (recordings .wav)
│       └── recordings/
│
├── src/                             # Code chính
│   ├── fraud_anomaly_detection.py   # Pipeline: Credit Card Fraud Detection
│   ├── gmm_speaker_identification.py# Pipeline: Speaker Identification
│   └── app.py                       # Ứng dụng Streamlit tương tác
│
├── notebook/                        # Jupyter Notebooks (phân tích chi tiết)
│   ├── fraud_anomaly_detection.ipynb
│   └── gmm_speaker_identification.ipynb
│
└── paper/                           # Bài báo khoa học
    └── paper.pdf
```

---

## 🔬 Bài toán 1 – Credit Card Fraud Detection

### Mô tả bài toán

Phát hiện gian lận trong **284,807 giao dịch thẻ tín dụng** của ngân hàng châu Âu (nguồn: ULB Machine Learning Group). Dữ liệu **cực kỳ mất cân bằng**: chỉ **0.17% (492 ca)** là gian lận thực sự.

### Thách thức

- Không thể dùng supervised learning thông thường do không có nhãn khi deploy
- Cần phương pháp **Anomaly Detection** thuần tuý (unsupervised)

### Phương pháp

```
Dữ liệu bình thường (99.83%)
         ↓
   Chuẩn hoá (StandardScaler)
         ↓
  Train GMM (K thành phần Gaussian)
         ↓
  Tính Log-Likelihood cho giao dịch mới
         ↓
  Log p(x) < ngưỡng → 🚨 GỬI CẢNH BÁO GIAN LẬN
```

### Kết quả

| Mô hình | ROC-AUC | Avg Precision | Recall (Fraud) |
|---------|---------|---------------|----------------|
| **GMM** | **cao hơn** | **cao hơn** | **tốt hơn** |
| K-Means | thấp hơn | thấp hơn | hạn chế |

> GMM vượt trội so với K-Means vì nó mô hình hóa **hình dạng xác suất thực sự** của dữ liệu thay vì chỉ dùng khoảng cách Euclidean.

---

## 🎙️ Bài toán 2 – Speaker Identification

### Mô tả bài toán

Nhận dạng **6 người nói** từ bộ dữ liệu **Free Spoken Digit Dataset (FSDD)**: 3,000 file `.wav` (8kHz) gồm các chữ số 0–9.

### Pipeline

```
File âm thanh (.wav, 8kHz)
         ↓
  Trích xuất MFCC (13 hệ số)
         ↓
  Kết hợp Delta + Delta² → 39 chiều
         ↓
  Train 1 GMM/speaker (K thành phần)
         ↓
  Nhận dạng: argmax  Log p(audio | GMM_speaker)
```

### Tại sao MFCC + GMM?

| Đặc trưng | Vai trò |
|-----------|---------|
| **MFCC** (13 chiều) | Mô phỏng cách tai người cảm nhận âm sắc |
| **Delta** (13 chiều) | Thể hiện sự biến đổi theo thời gian |
| **Delta²** (13 chiều) | Gia tốc biến đổi âm thanh |
| **GMM** | Học phân phối MFCC đặc trưng của từng giọng |

### Kết quả

- **Accuracy trên tập test**: > 90% với K = 12–16 thành phần Gaussian
- Mỗi speaker được mô hình hóa bằng **1 GMM riêng** với K Gaussian
- Nhận dạng: file audio nào cho **log-likelihood cao nhất** → đó là người nói

---

## 🧮 Lý thuyết GMM – EM Algorithm

GMM mô hình hóa dữ liệu bằng **hỗn hợp các phân phối Gaussian**:

$$p(x) = \sum_{c=1}^{K} \pi_c \cdot \mathcal{N}(x \mid \mu_c, \Sigma_c)$$

Tham số được ước lượng bằng thuật toán **Expectation-Maximization (EM)**:

**E-Step** – Tính trách nhiệm (soft assignment):
$$\gamma_{ic} = \frac{\pi_c \cdot \mathcal{N}(x_i \mid \mu_c, \Sigma_c)}{\sum_{k=1}^{K} \pi_k \cdot \mathcal{N}(x_i \mid \mu_k, \Sigma_k)}$$

**M-Step** – Cập nhật tham số:
$$\mu_c^{new} = \frac{\sum_i \gamma_{ic} \cdot x_i}{\sum_i \gamma_{ic}}, \quad
\Sigma_c^{new} = \frac{\sum_i \gamma_{ic}(x_i - \mu_c)(x_i - \mu_c)^T}{\sum_i \gamma_{ic}}$$

---

## 🚀 Hướng dẫn cài đặt & chạy

### 1. Clone repository

```bash
git clone https://github.com/<your-username>/GMM_Speaker_ID.git
cd GMM_Speaker_ID
```

### 2. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### 3. Tải dữ liệu

| Dữ liệu | Nguồn | Đặt vào |
|---------|-------|---------|
| `creditcard.csv` | [Kaggle – ULB Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) | `data/creditcard.csv` |
| FSDD recordings | [Kaggle – Free Spoken Digit Dataset](https://www.kaggle.com/datasets/joserzapata/free-spoken-digit-dataset-fsdd) | `data/archive/recordings/` |

> ⚠️ Các file dữ liệu có kích thước lớn nên **không** được đưa vào repository. Vui lòng tải về theo hướng dẫn trên.

### 4. Chạy Pipeline

**Fraud Detection:**
```bash
python src/fraud_anomaly_detection.py
```

**Speaker Identification:**
```bash
python src/gmm_speaker_identification.py
```

**Ứng dụng Streamlit tương tác:**
```bash
streamlit run src/app.py
```

### 5. Jupyter Notebooks

```bash
jupyter notebook notebook/
```

---

## 🖥️ Ứng dụng Streamlit – Demo Tương Tác

File `src/app.py` cung cấp giao diện web với **3 module**:

| Module | Chức năng |
|--------|-----------|
| 🧪 **GMM E-M Step-by-Step** | Minh họa từng bước toán học của thuật toán EM |
| 🎓 **Mô phỏng Huấn luyện ML** | Trực quan hoá quá trình train GMM & K-Means theo thời gian thực |
| 🎙️ **Speaker ID Inference** | Upload file `.wav` và nhận dạng người nói |

---

## 📦 Thư viện sử dụng

| Thư viện | Mục đích |
|----------|---------|
| `numpy`, `pandas` | Xử lý dữ liệu |
| `scikit-learn` | GMM, K-Means, PCA, Metrics |
| `librosa` | Trích xuất đặc trưng âm thanh (MFCC) |
| `matplotlib`, `seaborn` | Trực quan hoá |
| `streamlit` | Ứng dụng web tương tác |
| `scipy` | Phân phối xác suất |

---

## 👥 Tác giả

> Đây là bài báo thực hiện trong khuôn khổ môn học **Trí Tuệ Nhân Tạo / Machine Learning**.

---

## 📄 Tài liệu tham khảo

1. Dal Pozzolo, A. et al. (2015). *Calibrating Probability with Undersampling for Unbalanced Classification*. ULB Machine Learning Group.
2. Reynolds, D. A. (2009). *Gaussian Mixture Models*. Encyclopedia of Biometrics.
3. Davis, S. & Mermelstein, P. (1980). *Comparison of Parametric Representations for Monosyllabic Word Recognition*. IEEE TASLP.
4. [Free Spoken Digit Dataset (FSDD)](https://github.com/Jakobovski/free-spoken-digit-dataset)
5. [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

<div align="center">

⭐ Nếu dự án này có ích, hãy **star** repository nhé!

</div>
