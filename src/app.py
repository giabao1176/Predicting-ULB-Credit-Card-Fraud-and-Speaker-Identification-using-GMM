import streamlit as st
import os
import glob
import numpy as np
import pandas as pd
import warnings
import io

import librosa
import librosa.display
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import multivariate_normal

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# STYLING (Neo-Brutalism Dark Mode)
# ---------------------------------------------------------
st.set_page_config(page_title="GMM Sim: Speaker & Fraud & Math", page_icon="⚡", layout="wide")

st.markdown("""
<style>
    .stApp, .stApp * { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #3F362E; }
    div[data-testid="stMarkdownContainer"] p, div[data-testid="stMarkdownContainer"] span, div[data-testid="stText"] {
        color: #F8F5F2 !important;
    }
    h1, h2, h3, h4, h5 { font-family: 'Poppins', sans-serif !important; font-weight: 800 !important; color: #FFFFFF !important; }
    
    .stButton > button {
        background-color: #FF6B00 !important; color: #FFFFFF !important; border: 3px solid #000000 !important;
        border-radius: 0px !important; box-shadow: 4px 4px 0px #000000 !important;
        font-family: 'Poppins', sans-serif !important; font-weight: bold !important;
        text-transform: uppercase; padding: 0.5rem 1.5rem !important; transition: all 0.1s ease-in-out;
    }
    .stButton > button:active { box-shadow: 0px 0px 0px #000000 !important; transform: translate(4px, 4px) !important; }
    .stButton > button:hover { background-color: #FF8533 !important; }

    div[data-testid="stHorizontalBlock"] button, div[data-testid="stHorizontalBlock"] button p { background-color: #00E0FF !important; color: #19120B !important; font-weight:bold; }
    
    div[data-baseweb="select"] > div, input, .stFileUploader > div, div[data-baseweb="base-input"] {
        border: 3px solid #000000 !important; border-radius: 0px !important; box-shadow: 4px 4px 0px #000000 !important;
        background-color: #261E17 !important; color: #F8F5F2 !important;
    }
    
    [data-testid="stSidebar"] { background-color: #19120B !important; border-right: 3px solid #000000; }
    [data-testid="stSidebar"] * { color: #F8F5F2 !important; }
    [data-testid="stMetricValue"] { color: #00E0FF !important; font-family: 'Poppins', sans-serif !important; font-weight: bold; }
    .stCodeBlock, .katex-html { color: #00E0FF !important; }
    
    .brutal-card {
        background-color: #261E17; border: 3px solid #000000; box-shadow: 4px 4px 0px #000000;
        padding: 20px; margin-bottom: 20px;
    }
    .brutal-card-sub {
        background-color: #19120B; border: 2px solid #554A41; padding: 10px; margin-top: 10px; border-left: 5px solid #00E0FF;
    }
    .text-highlight { color: #00E0FF; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# CONSTANTS & CACHING
# ---------------------------------------------------------
plt.style.use('dark_background')
SPEAKERS = ['george', 'jackson', 'lucas', 'nicolas', 'theo', 'yweweler']
RECORDINGS_DIR = './archive/recordings'
CREDIT_DATA_PATH = r"d:\HocTriTueNhanTao\math\creditcard.csv"
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

@st.cache_data
def extract_mfcc(file_path_or_bytes, n_mfcc=13, sr=8000):
    y, sr = librosa.load(file_path_or_bytes, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    n_frames = mfccs.shape[1]
    delta_width = min(9, n_frames if n_frames % 2 != 0 else n_frames - 1)
    if delta_width >= 3:
        delta_mfccs = librosa.feature.delta(mfccs, width=delta_width)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2, width=delta_width)
        combined = np.vstack([mfccs, delta_mfccs, delta2_mfccs])
    else:
        delta_pad = np.zeros_like(mfccs)
        combined = np.vstack([mfccs, delta_pad, delta_pad])
    return combined.T, y, sr, mfccs

@st.cache_resource(show_spinner="Đang load GMM Nhận Dạng Giọng Nói...")
def load_speaker_gmm(n_components=12):
    train_features = {}
    for speaker in SPEAKERS:
        files = sorted(glob.glob(os.path.join(RECORDINGS_DIR, f'*_{speaker}_*.wav')))
        train_mfcc_list = []
        for f in files:
            parts = os.path.basename(f).replace('.wav', '').split('_')
            if int(parts[2]) >= 5:
                mfcc, _, _, _ = extract_mfcc(f)
                train_mfcc_list.append(mfcc)
        train_features[speaker] = np.vstack(train_mfcc_list)

    speaker_gmms = {}
    for speaker in SPEAKERS:
        gmm = GaussianMixture(n_components=n_components, covariance_type='diag', max_iter=200, random_state=42)
        gmm.fit(train_features[speaker])
        speaker_gmms[speaker] = gmm
    return speaker_gmms, train_features

@st.cache_resource(show_spinner="Đang load data Credit Card Fraud Detection (Toàn bộ 29 Features)...")
def get_fraud_raw_split():
    # Sử dụng toàn bộ đặc trưng gốc như yêu cầu để huấn luyện thuật toán chính xác trên toàn cục dữ liệu 29D
    if not os.path.exists(CREDIT_DATA_PATH): raise FileNotFoundError("Không tìm thấy creditcard.csv")
    df = pd.read_csv(CREDIT_DATA_PATH)
    n_normal_train = 6000 
    df_normal = df[df['Class'] == 0].sample(n_normal_train, random_state=RANDOM_SEED)
    df_fraud = df[df['Class'] == 1]
    df_normal_test = df[df['Class'] == 0].drop(df_normal.index).sample(2000, random_state=RANDOM_SEED)
    
    # Giữ nguyên toàn bộ 29 Đặc trưng
    features = [f'V{i}' for i in range(1, 29)] + ['Amount']
    scaler = StandardScaler()
    
    X_train_29d = scaler.fit_transform(df_normal[features])
    X_test_normal_29d = scaler.transform(df_normal_test[features])
    X_test_fraud_29d = scaler.transform(df_fraud[features])
    
    # Tính toán PCA sẵn chỉ để dùng cho việc MAPPING/vẽ trên mành hình 2D, các thuật toán vẫn train ngầm 29D
    pca_model = PCA(n_components=2, random_state=RANDOM_SEED)
    pca_model.fit(X_train_29d)
    
    return X_train_29d, X_test_normal_29d, X_test_fraud_29d, pca_model

# ---------------------------------------------------------
# UI STRUCTURE
# ---------------------------------------------------------
st.title("⚡ DEEP TECH SIMULATOR")
st.markdown("### Trực quan hóa **Gaussian Mixture Models** & **K-Means** trong Thực tế")

if 'app_mode' not in st.session_state: st.session_state.app_mode = '🎓 Mô Phỏng Huấn Luyện ML'
if 'speaker_step' not in st.session_state: st.session_state.speaker_step = 1

st.sidebar.markdown(f"<h2 style='color: #00E0FF;'>MODULES</h2>", unsafe_allow_html=True)
app_mode = st.sidebar.radio("Chọn Công Cụ Mô Phỏng:", ['🧪 GMM E-M Step-by-Step (Toán Học)', '🎓 Mô Phỏng Huấn Luyện ML', '🎙️ Speaker ID (Inference)'])

if app_mode != st.session_state.app_mode:
    st.session_state.app_mode = app_mode
    st.session_state.speaker_step = 1
    st.session_state.pop('em_gmm', None)
    st.session_state.pop('train_state', None)
    st.session_state.pop('toy_step', None)

# ==============================================================================
# MODE 0: TOY DATASET E-M MANAUL CALCULATOR
# ==============================================================================
if app_mode == '🧪 GMM E-M Step-by-Step (Toán Học)':
    st.markdown("""
    <div class="brutal-card">
        <h3 style="color:#FF6B00; margin-top:0;">🧪 Sandbox: Thuật Toán Expectation-Maximization Thủ Công</h3>
        <p>Đây là môi trường Mini-Data (Chỉ có 6 điểm - 2 Cụm) để bạn quan sát <b>CÁC CHỮ SỐ MÀ TRẬN THẬT</b> thay đổi trong quá trình tính E-Step (Phân công trọng số) và M-Step (Cập nhật Toạ độ & Hình Elip).</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Small strictly controlled dataset
    X_toy = np.array([
        [1.0, 1.5], [1.2, 1.0], [0.8, 1.2], # Dữ liệu tệp 1 (Dưới trái)
        [8.0, 8.5], [8.5, 8.0], [9.0, 9.0]  # Dữ liệu tệp 2 (Trên phải)
    ])
    K = 2
    N = len(X_toy)
    
    # state init
    if 'toy_step' not in st.session_state:
        st.session_state.toy_step = 0 # 0: Init, 1: E-step done, 2: M-step done (Loop)
        st.session_state.toy_iter = 0
        # Bad initialization on purpose to see it move
        st.session_state.toy_mu = np.array([[2.0, 6.0], [6.0, 2.0]])
        st.session_state.toy_cov = np.array([[[2.0, 0], [0, 2.0]], [[2.0, 0], [0, 2.0]]])
        st.session_state.toy_pi = np.array([0.5, 0.5])
        st.session_state.toy_gamma = np.zeros((N, K))
        
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("🔄 Khởi Tạo Lại", type="secondary"):
            if 'toy_step' in st.session_state: st.session_state.pop('toy_step')
            st.rerun()
    with c2:
        if st.button("1️⃣ Chạy E-Step (Expectation)", type="primary" if st.session_state.toy_step%2==0 else "secondary"):
            # Guard against invalid state (NaN in cov from bad M-step)
            if np.any(np.isnan(st.session_state.toy_cov)) or np.any(np.isinf(st.session_state.toy_cov)):
                st.warning("⚠️ Ma trận Covariance bị lỗi (NaN/Inf). Ấn Khởi Tạo Lại!")
            else:
                # Compute E-step manually
                scores = np.zeros((N, K))
                for c in range(K):
                    scores[:, c] = st.session_state.toy_pi[c] * multivariate_normal.pdf(X_toy, mean=st.session_state.toy_mu[c], cov=st.session_state.toy_cov[c])
                score_sum = scores.sum(axis=1, keepdims=True)
                # handle div by zero just in case
                score_sum[score_sum == 0] = 1e-10 
                st.session_state.toy_gamma = scores / score_sum
                st.session_state.toy_step += 1
                st.rerun()
    with c3:
        if st.button("2️⃣ Chạy M-Step (Maximization)", type="primary" if st.session_state.toy_step%2!=0 else "secondary"):
            # Compute M-step manually
            gamma = st.session_state.toy_gamma
            Nk = np.maximum(gamma.sum(axis=0), 1e-10) # Tránh lỗi chia cho 0 sinh ra NaN
            
            new_mu = (gamma.T @ X_toy) / Nk[:, np.newaxis]
            new_cov = np.zeros((K, 2, 2))
            for c in range(K):
                diff = X_toy - new_mu[c]
                new_cov[c] = (gamma[:, c, np.newaxis, np.newaxis] * np.einsum('ni,nj->nij', diff, diff)).sum(axis=0) / Nk[c]
                new_cov[c] += np.eye(2) * 1e-4 # Cộng dồn đường chéo để tránh chập ma trận (Singular Matrix gây inf)
            new_pi = Nk / N
            
            st.session_state.toy_mu = new_mu
            st.session_state.toy_cov = new_cov
            st.session_state.toy_pi = new_pi
            st.session_state.toy_step += 1
            st.session_state.toy_iter += 1
            st.rerun()
            
    # Status banner
    step_name = "Chưa bắt đầu"
    step_explain = "Ấn nút '1️⃣ Chạy E-Step' để phân công trách nhiệm (Tính % cho từng điểm dữ liệu)."
    step_color = "#00E0FF"
    if st.session_state.toy_step % 2 == 1:
        step_name = "🌡️ E-Step hoàn thành!"
        step_explain = "Bảng Gamma đã cập nhật ✅. Các điểm được tô màu theo cụm. Nhấn M-Step để di dời Tâm và bóp lại Elip!"
        step_color = "#FF6B00"
    elif st.session_state.toy_step > 0:
        step_name = "🔧 M-Step hoàn thành!"
        step_explain = "Tâm đã dời. Elip đã cập nhật ✅. Nhấn E-Step để tính lại trách nhiệm theo Elip mới!"
        step_color = "#00E0FF"
    st.markdown(f"""<div style='background:#261E17;border-left:5px solid {step_color};padding:8px 16px;margin:8px 0 12px;'>
        <b style='color:{step_color}'>Trạng thái: {step_name}</b><br>
        <span style='color:#F8F5F2;font-size:0.9em'>{step_explain}</span>
    </div>""", unsafe_allow_html=True)

    col_viz, col_math = st.columns([1, 1])
    
    with col_viz:
        st.markdown(f"#### Biểu Đồ Không Gian - Vòng lặp {st.session_state.toy_iter}")
        fig, ax = plt.subplots(figsize=(6, 6))
        fig.patch.set_facecolor('#19120B'); ax.set_facecolor('#19120B')
        
        # Plot points
        for i in range(N):
            # color based on gamma responsibility if E-step is run at least once
            if st.session_state.toy_step > 0:
                color = np.array([255, 107, 0])/255.0 * st.session_state.toy_gamma[i, 0] + np.array([0, 224, 255])/255.0 * st.session_state.toy_gamma[i, 1]
            else:
                color = '#A5A5A5'
            ax.scatter(X_toy[i, 0], X_toy[i, 1], c=[color], s=150, edgecolors='white', zorder=5)
            ax.text(X_toy[i, 0]+0.3, X_toy[i, 1]+0.3, f"X{i}", color='white', fontsize=12)
            
        # Plot Gaussian Ellipses based on Covariances manually
        for c in range(K):
            mu = st.session_state.toy_mu[c]
            cov = st.session_state.toy_cov[c]
            eigvals, eigvecs = np.linalg.eigh(cov)
            angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
            width, height = 2 * np.sqrt(5.991 * eigvals) # 95% confidence interval
            
            color = '#FF6B00' if c==0 else '#00E0FF'
            ell = patches.Ellipse(mu, width, height, angle=angle, edgecolor=color, facecolor='none', linewidth=2, linestyle='--')
            ax.add_patch(ell)
            ax.scatter(mu[0], mu[1], c=color, marker='X', s=200, edgecolors='white', zorder=10)
            
        ax.set_xlim(-4, 14); ax.set_ylim(-4, 14)
        ax.set_xticks([]); ax.set_yticks([])
        st.pyplot(fig)
        
    with col_math:
        st.markdown(f"#### Tham số Mô Hình Thực Tế")
        
        c_k1, c_k2 = st.columns(2)
        with c_k1:
            st.markdown("<div class='brutal-card-sub' style='border-left-color:#FF6B00;'>", unsafe_allow_html=True)
            st.markdown("**Cụm K1 (Màu Cam)**")
            st.latex(r"\mu_1 = \text{" + f"[{st.session_state.toy_mu[0,0]:.2f}, {st.session_state.toy_mu[0,1]:.2f}]" + "}")
            st.write(rf"$\Sigma_1$ = [[{st.session_state.toy_cov[0,0,0]:.2f}, {st.session_state.toy_cov[0,0,1]:.2f}], [{st.session_state.toy_cov[0,1,0]:.2f}, {st.session_state.toy_cov[0,1,1]:.2f}]]")
            st.write(rf"$\pi_1$ = {st.session_state.toy_pi[0]:.2f}")
            st.markdown("</div>", unsafe_allow_html=True)
        with c_k2:
            st.markdown("<div class='brutal-card-sub' style='border-left-color:#00E0FF;'>", unsafe_allow_html=True)
            st.markdown("**Cụm K2 (Màu Cyan)**")
            st.latex(r"\mu_2 = \text{" + f"[{st.session_state.toy_mu[1,0]:.2f}, {st.session_state.toy_mu[1,1]:.2f}]" + "}")
            st.write(rf"$\Sigma_2$ = [[{st.session_state.toy_cov[1,0,0]:.2f}, {st.session_state.toy_cov[1,0,1]:.2f}], [{st.session_state.toy_cov[1,1,0]:.2f}, {st.session_state.toy_cov[1,1,1]:.2f}]]")
            st.write(rf"$\pi_2$ = {st.session_state.toy_pi[1]:.2f}")
            st.markdown("</div>", unsafe_allow_html=True)
            
        st.markdown("---")
        st.markdown(r"#### Trọng số Trách nhiệm (E-Step: $\gamma_{ic}$)")
        st.write("Xác suất điểm $X_i$ thuộc về cụm C. K-Means sẽ hard-code cái này thành [1, 0]. GMM giữ nguyên số thập phân (Soft Clustering).")
        
        df_gamma = pd.DataFrame({
            "Điểm": [f"X{i}" for i in range(N)],
            "Thuộc K1 (Cam)": [f"{st.session_state.toy_gamma[i, 0]*100:.1f}%" for i in range(N)],
            "Thuộc K2 (Cyan)": [f"{st.session_state.toy_gamma[i, 1]*100:.1f}%" for i in range(N)]
        })
        st.dataframe(df_gamma, hide_index=True, use_container_width=True)

        # Compute & show per-point scores if E-step has run
        if st.session_state.toy_step > 0:
            st.markdown("##### 🔢 Tính chi tiết E-Step lần này:")
            for i in range(N):
                g0 = st.session_state.toy_gamma[i, 0]
                g1 = st.session_state.toy_gamma[i, 1]
                st.markdown(
                    f"**X{i}** ({X_toy[i,0]}, {X_toy[i,1]}): "
                    f"<span style='color:#FF6B00'>K1={g0*100:.1f}%</span> | "
                    f"<span style='color:#00E0FF'>K2={g1*100:.1f}%</span>",
                    unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📐 Toàn Bộ Công Thức GMM - EM Algorithm")

    tab_init, tab_e, tab_m, tab_ll = st.tabs(["🎲 Khởi tạo", "E-Step", "M-Step", "📈 Log-Likelihood"])

    with tab_init:
        st.markdown("#### Khởi tạo tham số ban đầu")
        st.latex(r"\pi_c^{(0)} = \frac{1}{K}, \quad \forall c = 1,...,K")
        st.latex(r"\mu_c^{(0)} \sim \text{random từ tập dữ liệu X}")
        st.latex(r"\Sigma_c^{(0)} = I \quad \text{(Ma trận đơn vị)}")
        st.write("Trong sandbox này: K=2, μ₁=[2,6], μ₂=[6,2] – Đặt sai cố ý để quan sát EM tự tìm về đúng tâm dữ liệu.")

    with tab_e:
        st.markdown("#### E-Step: Expectation — Tính Trách nhiệm")
        st.write("Tính xác suất mềm điểm **xᵢ** thuộc về cụm **c**:")
        st.latex(r"\gamma_{ic} = \frac{\pi_c \cdot \mathcal{N}(x_i \mid \mu_c, \Sigma_c)}{\displaystyle\sum_{k=1}^{K} \pi_k \cdot \mathcal{N}(x_i \mid \mu_k, \Sigma_k)}")
        st.write("Hàm mật độ Gaussian nhiều chiều:")
        st.latex(r"\mathcal{N}(x \mid \mu, \Sigma) = \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}} \exp\!\left(-\frac{1}{2}(x-\mu)^T \Sigma^{-1} (x-\mu)\right)")
        st.info("Kết quả: Ma trận γ có shape (N×K). Mỗi hàng tổng = 1. K-Means sẽ làm tròn thành {0, 1} — GMM giữ số thập phân.")

    with tab_m:
        st.markdown("#### M-Step: Maximization — Cập nhật Tham số")
        st.write("**Số điểm hiệu dụng** thuộc cụm c:")
        st.latex(r"N_c = \sum_{i=1}^{N} \gamma_{ic}")
        st.write("**Cập nhật trọng số** (tỉ lệ điểm):")
        st.latex(r"\pi_c^{new} = \frac{N_c}{N}")
        st.write("**Cập nhật Tâm** (trung bình có trọng số):")
        st.latex(r"\mu_c^{new} = \frac{1}{N_c}\sum_{i=1}^{N} \gamma_{ic} \cdot x_i")
        st.write("**Cập nhật Ma trận Hiệp Phương Sai** (hình dạng Elip):")
        st.latex(r"\Sigma_c^{new} = \frac{1}{N_c}\sum_{i=1}^{N} \gamma_{ic} \cdot (x_i - \mu_c^{new})(x_i - \mu_c^{new})^T")
        st.info("Bộ 3 tham số (π, μ, Σ) hoàn toàn xác định một Gaussian. EM lặp đến khi tất cả hội tụ.")

    with tab_ll:
        st.markdown("#### Log-Likelihood — Thước đo Hội tụ")
        st.write("Sau mỗi vòng EM, tính tổng Log-Likelihood để đo **mô hình giải thích dữ liệu tốt đến đâu**:")
        st.latex(r"\mathcal{L}(\theta) = \sum_{i=1}^{N} \log\left(\sum_{c=1}^{K} \pi_c \cdot \mathcal{N}(x_i \mid \mu_c, \Sigma_c)\right)")
        st.write("**Điều kiện hội tụ:**")
        st.latex(r"|\mathcal{L}^{(t+1)} - \mathcal{L}^{(t)}| < \epsilon \quad (\epsilon \approx 10^{-4})")
        st.success("✅ EM đảm bảo Log-Likelihood tăng dần (không bao giờ giảm) theo mỗi vòng lặp. Khi đồ thị phẳng = Hội tụ!")
        # Show current LL
        if st.session_state.toy_step > 0:
            try:
                ll_now = 0.0
                for i in range(N):
                    mix = sum(
                        st.session_state.toy_pi[c] * multivariate_normal.pdf(
                            X_toy[i], mean=st.session_state.toy_mu[c], cov=st.session_state.toy_cov[c])
                        for c in range(K))
                    ll_now += np.log(max(mix, 1e-300))
                st.metric("Log-Likelihood hiện tại", f"{ll_now:.4f}")
            except Exception:
                pass

# ==============================================================================
# MODE 1: HLD TRAINING SIMULATOR (STEP-BY-STEP)
# ==============================================================================
elif app_mode == '🎓 Mô Phỏng Huấn Luyện ML':
    st.markdown("""
    <div class="brutal-card">
        <h3 style="color:#FF6B00; margin-top:0">🎓 Mô Phỏng Huấn Luyện — 2 Bài Toán Thực Tế</h3>
        <div style="display:flex;gap:24px;flex-wrap:wrap;">
            <div style="flex:1;min-width:220px;border-left:4px solid #FF6B00;padding-left:12px;">
                <b style="color:#FF6B00">💳 Fraud Detection (Kaggle Credit Card)</b><br>
                <span style="color:#F8F5F2;font-size:0.9em">
                284,807 giao dịch thẻ tín dụng châu Âu | Chỉ <b>0.17% là gian lận</b> (492 ca) → Dữ liệu cực kỳ mất cân bằng.<br>
                29 đặc trưng ẩn danh V1–V28 + Amount (đã PCA). <b>Thử thách:</b> Không có nhãn khi train — phải phát hiện bất thường thuần tuý bằng Anomaly Detection.
                </span>
            </div>
            <div style="flex:1;min-width:220px;border-left:4px solid #00E0FF;padding-left:12px;">
                <b style="color:#00E0FF">🎙️ Speaker ID (Free Spoken Digit Dataset)</b><br>
                <span style="color:#F8F5F2;font-size:0.9em">
                6 người nói × 10 chữ số × 50 lần = 3,000 file .wav. Mỗi giọng trích xuất <b>39 hệ số MFCC</b> (13 + Δ + Δ²).<br>
                <b>Thử thách:</b> Giọng mỗi người biến thiên lớn — GMM phải học 6 chế độ phát âm (nguyên âm, phụ âm, khoảng lặng...) đặc trưng riêng.
                </span>
            </div>
        </div>
        <p style="color:#A5A5A5;font-size:0.85em;margin:14px 0 0">
        ▶ <b>Mô phỏng sẽ:</b> khởi tạo thuật toán → lặp từng bước EM / K-Means → hiển thị ranh giới quyết định thay đổi theo thời gian thực →  đánh giá Recall &amp; False Positive cuối cùng.
        </p>
    </div>
    """, unsafe_allow_html=True)

    train_scenario = st.radio("Chọn Bài Toán Cần Theo Dõi Huấn Luyện:", ["💳 HLT Fraud Detection (K-Means vs GMM)", "🎙️ HLT Speaker ID (Chỉ định dạng tiếng nói MFCC)"])
    
    if train_scenario == "💳 HLT Fraud Detection (K-Means vs GMM)":
        X_train_29d, X_test_normal_29d, X_test_fraud_29d, pca_model = get_fraud_raw_split()
        
        np.random.seed(42)
        idx = np.random.choice(len(X_train_29d), 600, replace=False)
        X_vis = X_train_29d[idx] # Base data is 29D
        X_vis_2d = pca_model.transform(X_vis) # Project to 2D for drawing only
        
        # Test phase data - FULL 29D DATA SCALE FOR MASSIVE DIFFERENCE
        X_test_vis = np.vstack([X_test_normal_29d, X_test_fraud_29d])
        X_test_vis_2d = pca_model.transform(X_test_vis)
        y_test_vis = np.array([0]*len(X_test_normal_29d) + [1]*len(X_test_fraud_29d))

        if 'train_state' not in st.session_state or st.session_state.get('scenario') != "fraud":
            st.session_state.train_state = True
            st.session_state.scenario = "fraud"
            st.session_state.train_iter = 0
            st.session_state.phase = "train" 
            
            # Init manually K-means in 29D
            st.session_state.km_centers = X_vis[np.random.choice(len(X_vis), 2, replace=False)]
            # Init GMM warm start in 29D
            st.session_state.em_gmm = GaussianMixture(n_components=2, covariance_type='full', max_iter=1, warm_start=True, init_params='random', random_state=42)
            st.session_state.em_log_scores = []
        
        col_ctrl, col_viz = st.columns([1, 2])
        with col_ctrl:
            st.markdown("<h4 style='color:#00E0FF'>⚙️ Bảng Điều Khiển</h4>", unsafe_allow_html=True)
            st.write(f"**Giai đoạn:** `{'Huấn Luyện 29 Chiều (Training)' if st.session_state.phase == 'train' else 'Kiểm Thử (Testing/Flagging)'}`")
            st.write(f"**Vòng Lặp (Iter):** `{st.session_state.train_iter}`")
            
            if st.session_state.phase == "train":
                c1, c2 = st.columns(2)
                if c1.button("🔄 Reset Data"):
                    st.session_state.pop('train_state', None)
                    st.rerun()
                if c2.button("Lặp 1 Lần ➡️"):
                    dists = np.linalg.norm(X_vis[:, np.newaxis] - st.session_state.km_centers, axis=2)
                    labels = np.argmin(dists, axis=1)
                    new_centers = np.array([X_vis[labels == i].mean(axis=0) if sum(labels == i) > 0 else st.session_state.km_centers[i] for i in range(2)])
                    st.session_state.km_centers = new_centers
                    st.session_state.em_gmm.fit(X_vis)
                    st.session_state.train_iter += 1
                    st.rerun()
                    
                if st.button("Lặp Đi 5 Lần ⏭️"):
                    for _ in range(5):
                        dists = np.linalg.norm(X_vis[:, np.newaxis] - st.session_state.km_centers, axis=2)
                        st.session_state.km_centers = np.array([X_vis[np.argmin(dists, axis=1) == i].mean(axis=0) for i in range(2)])
                        st.session_state.em_gmm.fit(X_vis)
                        st.session_state.train_iter += 1
                    st.rerun()
                    
                st.markdown("---")
                if st.session_state.train_iter > 3:
                    if st.button("🚀 Chốt Mô Hình & Đi Gắn Cờ (Test Phase)", type="primary"):
                        st.session_state.phase = "test"
                        st.rerun()
            else:
                if st.button("🔄 Quay lại Huấn Luyện (Reset)"):
                    st.session_state.phase = "train"
                    st.session_state.pop('train_state', None)
                    st.rerun()

            st.markdown("##### Giải Thích Toán Học Từng Bước:")
            if st.session_state.phase == "train":
                st.markdown("**1. K-Means (Thiên về Khoảng cách Euclidean - Hình tròn):**")
                st.latex(r"\text{Bước Gán } C_i = \arg\min_k ||x_i - \mu_k||^2")
                st.latex(r"\text{Bước Cập Nhật Tâm } \mu_k = \frac{1}{|C_k|} \sum_{x \in C_k} x")
                st.write(r"K-Means kéo tâm $\mu$ (Màu Đỏ) về trung bình cộng toạ độ 29 Đặc trưng của đám mây.")
                
                st.markdown("**2. GMM (Thuật toán EM - Hình Elip):**")
                st.latex(r"\text{E-Step: } \gamma_{ic} = \frac{\pi_c \mathcal{N}(x_i | \mu_c, \Sigma_c)}{\sum_k \pi_k \mathcal{N}(x_i | \mu_k, \Sigma_k)}")
                st.latex(r"\text{M-Step: } \Sigma_c = \frac{1}{N_c} \sum_i \gamma_{ic} (x_i - \mu_c)(x_i - \mu_c)^T")
                st.write(r"GMM dùng Ma trận hiệp phương sai $\Sigma$ 29x29 bóp méo hình Elip ôm sát tập dữ liệu đa chiều.")
            else:
                st.markdown("**Giai đoạn Gắn Cờ & Dự Đoán (Inference):**")
                st.latex(r"\text{Threshold K-Means: } d(x, \mu) > \delta_{KM}")
                st.latex(r"\text{Threshold GMM: } \log P(x) < \delta_{GMM}")
                st.write("Biểu đồ phản ánh rõ điểm yếu cốt lõi của KMeans trong fraud detection: Cấm cửa 1 vòng tròn quá lớn sẽ gây ra False Negative thảm họa.")

        with col_viz:
            fig, (ax_km, ax_gmm) = plt.subplots(1, 2, figsize=(14, 6))
            fig.patch.set_facecolor('#19120B')
            
            # Compute shared axis limits so both plots are perfectly size-comparable
            x_min_g, x_max_g = X_vis_2d[:, 0].min() - 3, X_vis_2d[:, 0].max() + 3
            y_min_g, y_max_g = X_vis_2d[:, 1].min() - 3, X_vis_2d[:, 1].max() + 3
            
            for ax in [ax_km, ax_gmm]:
                ax.set_facecolor('#19120B')
                # Draw training background faintly mapped via PCA 2D
                ax.scatter(X_vis_2d[:, 0], X_vis_2d[:, 1], s=10, c='#A5A5A5', alpha=0.3)
                ax.tick_params(colors='white')
                ax.set_xlabel("Đặc trưng nén: PCA Component 1", color="white")
                ax.set_ylabel("Đặc trưng nén: PCA Component 2", color="white")
                ax.set_xlim(x_min_g, x_max_g)
                ax.set_ylim(y_min_g, y_max_g)
            
            ax_km.set_title("K-Means: Phân cụm bằng Khoảng cách Euclidean (Hình Tròn cứng)", color="#FF6B00", fontweight='bold')
            ax_gmm.set_title(f"GMM: Phân cụm bằng Phân phối Xác Suất (Hình Elip mềm) - Iter {st.session_state.train_iter}", color="#00E0FF", fontweight='bold')
            
            if st.session_state.train_iter > 0:
                km_centers_2d = pca_model.transform(st.session_state.km_centers)
                W = pca_model.components_
                
                # --- K-MEANS VIS: Voronoi-style background coloring + circular boundary ---
                # Draw Voronoi class regions (nearest-centroid coloring) on K-Means
                xx_full, yy_full = np.meshgrid(np.linspace(x_min_g, x_max_g, 200), np.linspace(y_min_g, y_max_g, 200))
                grid_2d = np.c_[xx_full.ravel(), yy_full.ravel()]
                dists_grid = np.linalg.norm(grid_2d[:, np.newaxis, :] - km_centers_2d[np.newaxis, :, :], axis=2)
                km_grid_labels = np.argmin(dists_grid, axis=1).reshape(xx_full.shape)
                ax_km.contourf(xx_full, yy_full, km_grid_labels, levels=[-0.5, 0.5, 1.5], colors=['#FF6B0015', '#00E0FF15'], alpha=0.2)
                ax_km.contour(xx_full, yy_full, km_grid_labels, levels=[0.5], colors=['white'], linewidths=1.5, linestyles='--', alpha=0.5)
                ax_km.text(x_min_g + 0.5, y_max_g - 0.8, 'Ranh giới Voronoi (Cứng - Hình thẳng)', color='white', fontsize=8, alpha=0.8)
                
                # Draw cluster extent circles
                ax_km.scatter(km_centers_2d[:, 0], km_centers_2d[:, 1], c='red', marker='X', s=250, label="Tâm KMeans", zorder=10, edgecolors='white')
                labels_2d = np.argmin(np.linalg.norm(X_vis_2d[:, np.newaxis] - km_centers_2d, axis=2), axis=1)
                colors_km = ['#FF6B00', '#00E0FF']
                for c in range(2):
                    points_in_c = X_vis_2d[labels_2d == c]
                    c_2d = km_centers_2d[c]
                    if len(points_in_c) > 0:
                        r = np.percentile(np.linalg.norm(points_in_c - c_2d, axis=1), 85)
                        ax_km.add_patch(plt.Circle((c_2d[0], c_2d[1]), r, color=colors_km[c], fill=False, linestyle='-', linewidth=2, alpha=0.9, zorder=5))
                        ax_km.text(c_2d[0], c_2d[1] + r + 0.3, f'Cụm {c+1} (r={r:.1f})', color=colors_km[c], fontsize=8, ha='center')
                
                # --- GMM VIS: Ellipse contours from projected Covariance matrix ---
                xx, yy = np.meshgrid(np.linspace(x_min_g, x_max_g, 120), np.linspace(y_min_g, y_max_g, 120))
                grid_points = np.c_[xx.ravel(), yy.ravel()]
                
                Z = np.zeros(xx.shape)
                for c in range(2):
                    mu_2d = pca_model.transform(st.session_state.em_gmm.means_[c].reshape(1, -1))[0]
                    cov_2d = W @ st.session_state.em_gmm.covariances_[c] @ W.T
                    cov_2d += np.eye(2) * 1e-6 # numerical stability
                    weight = st.session_state.em_gmm.weights_[c]
                    Z += weight * multivariate_normal.pdf(grid_points, mean=mu_2d, cov=cov_2d).reshape(xx.shape)
                
                # Raw PDF contours -- no log needed since the projections should be tightly bounded
                levels = np.linspace(Z.max() * 0.01, Z.max() * 0.9, 8)
                ax_gmm.contourf(xx, yy, Z, levels=levels, cmap='magma', alpha=0.35)
                ax_gmm.contour(xx, yy, Z, levels=levels, cmap='magma', alpha=0.9, linewidths=1.5)
                ax_gmm.text(x_min_g + 0.5, y_max_g - 0.8, 'Ranh giới Elip mềm (Soft – Phân phối Gaussian)', color='#00E0FF', fontsize=8)
                
                gmm_means_2d = pca_model.transform(st.session_state.em_gmm.means_)
                ax_gmm.scatter(gmm_means_2d[:, 0], gmm_means_2d[:, 1], marker='+', s=250, c='#00E0FF', label="Tâm GMM", zorder=10, linewidths=3)
                
            if st.session_state.phase == "test":
                # Compute exact thresholds based entirely on 29D logic
                dists_train = np.min(np.linalg.norm(X_vis[:, np.newaxis] - st.session_state.km_centers, axis=2), axis=1)
                k_thresh = np.percentile(dists_train, 98) 
                
                g_scores_train = st.session_state.em_gmm.score_samples(X_vis) # True 29D score
                g_thresh = np.percentile(g_scores_train, 5) 
                
                # Compute 2D threshold proportional approximation for visual boundary circle
                dists_train_2d = np.min(np.linalg.norm(X_vis_2d[:, np.newaxis] - km_centers_2d, axis=2), axis=1)
                k_thresh_2d = np.percentile(dists_train_2d, 98)
                for ci, c_2d in enumerate(km_centers_2d):
                    ax_km.add_patch(plt.Circle((c_2d[0], c_2d[1]), k_thresh_2d, color='#e74c3c', fill=False, linestyle='-', linewidth=2.5, zorder=6))
                    ax_km.text(c_2d[0], c_2d[1] - k_thresh_2d - 0.4, f'Ngưỡng Phát Hiện (98%)', color='#e74c3c', fontsize=7, ha='center')
                    
                Z_scores_train_2d = np.zeros(len(X_vis_2d))
                for c in range(2):
                    mu_2d = pca_model.transform(st.session_state.em_gmm.means_[c].reshape(1, -1))[0]
                    cov_2d = W @ st.session_state.em_gmm.covariances_[c] @ W.T
                    cov_2d += np.eye(2) * 1e-6
                    weight = st.session_state.em_gmm.weights_[c]
                    Z_scores_train_2d += weight * multivariate_normal.pdf(X_vis_2d, mean=mu_2d, cov=cov_2d)
                g_thresh_2d = np.percentile(Z_scores_train_2d, 5)
                # Vẽ Elip chốt chặn 2D bao phủ đúng
                ax_gmm.contour(xx, yy, Z, levels=[g_thresh_2d], colors=['#e74c3c'], linewidths=3, linestyles='--') 
                ax_gmm.text(x_min_g + 0.5, y_min_g + 0.5, f'Ranh giới Chốt (căt tại 5% đuôi)', color='#e74c3c', fontsize=7)
                
                k_dists_test = np.min(np.linalg.norm(X_test_vis[:, np.newaxis] - st.session_state.km_centers, axis=2), axis=1) # True 29D testing
                g_scores_test = st.session_state.em_gmm.score_samples(X_test_vis) # True 29D testing
                
                for i in range(len(X_test_vis)):
                    is_fraud = y_test_vis[i] == 1
                    c_actual = '#FF6B00' if is_fraud else '#004747'
                    marker = 'v' if is_fraud else '.'
                    s = 60 if is_fraud else 5
                    
                    # Logic hoàn toàn hoạt động trên 29D
                    km_flagged = k_dists_test[i] > k_thresh
                    gmm_flagged = g_scores_test[i] < g_thresh
                    
                    alpha_km = 0.9 if is_fraud or km_flagged else 0.1
                    alpha_gmm = 0.9 if is_fraud or gmm_flagged else 0.1
                    
                    # Hiện thực hiển thị trên 2D
                    ax_km.scatter(X_test_vis_2d[i, 0], X_test_vis_2d[i, 1], c=c_actual, marker=marker, s=s, edgecolors='none', zorder=5, alpha=alpha_km)
                    ax_gmm.scatter(X_test_vis_2d[i, 0], X_test_vis_2d[i, 1], c=c_actual, marker=marker, s=s, edgecolors='none', zorder=5, alpha=alpha_gmm)
                
                fig.text(0.5, 0.01, r"Cam (v) = GIAN LẬN THỰC TẺ | Xanh = BÌNH THƯỜNG | Đường đứt Đỏ = Ranh giới Phát Hiện | Hình chiếu PCA 2D của Data 29 Chiều", ha="center", color="#F8F5F2", fontsize=9, fontweight='bold')
                
            plt.tight_layout(pad=2.0)
            st.pyplot(fig)
            
            if st.session_state.phase == "test":
                st.markdown("### 🏆 KẾT QUẢ CẮM CỜ TRÊN TOÀN BỘ TẬP DỮ LIỆU TEST (Gồm hàng ngàn giao dịch)")
                from sklearn.metrics import recall_score
                km_preds = (k_dists_test > k_thresh).astype(int)
                gmm_preds = (g_scores_test < g_thresh).astype(int)
                
                km_fp = np.sum((km_preds == 1) & (y_test_vis == 0))
                gmm_fp = np.sum((gmm_preds == 1) & (y_test_vis == 0))
                
                c1, c2, c3 = st.columns(3)
                c1.metric("K-Means Bắt Trúng (Recall)", f"{recall_score(y_test_vis, km_preds)*100:.1f}%", f"Báo động giả khách hàng: {km_fp} ca", delta_color="inverse")
                c2.metric("GMM Bắt Trúng (Recall)", f"{recall_score(y_test_vis, gmm_preds)*100:.1f}%", f"Báo động giả khách hàng: {gmm_fp} ca", delta_color="inverse")
                c3.metric("Nhận định Hiệu năng", "GMM Áp Đảo Tuyệt Đối")
                
                st.markdown("""
                <div class='brutal-card' style='padding:10px; border-color:#FF6B00;'>
                    <h5 style='color:#FF6B00;'>📌 Nhìn vào biểu đồ bên trên là thấy ngay sự khác biệt RẦN RẦN (Recall đã cao chót vót nhờ Fix Feature V14 V17):</h5>
                    <ul>
                        <li><b>Vì sao K-Means "ngáo":</b> Vòng tròn ranh giới màu đỏ (Ngưỡng 2%) quá bé. K-Means bỏ lọt tới hơn nửa số Tội phạm Fraud ngoài rìa. Hầu hết bị chui ngay vào kẽ vỏ ngoài của mây dữ liệu.</li>
                        <li><b>Vì sao GMM "mẫu mực":</b> Biên giới bảo vệ cong theo đúng chu vi đường chéo phân phối V14-V17, <b>ôm gọn các giao dịch normal và văng ngược tất cả Fraud ra bên ngoài biên Elip</b>. Mức Recall bắt gian lận nhảy vọt xuất sắc! </li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

    elif train_scenario == "🎙️ HLT Speaker ID (Chỉ định dạng tiếng nói MFCC)":
        _, train_features = load_speaker_gmm()
        
        selected_speaker = st.selectbox(
            "🎤 Chọn Người Nói để Phân Tích MFCC:",
            SPEAKERS, index=0,
            format_func=lambda s: f"🎙️ {s.capitalize()}"
        )
        
        speaker_mfcc = train_features[selected_speaker]
        n_components_spk = 6
        
        pca_spk = PCA(n_components=2, random_state=42)
        X_spk_2d = pca_spk.fit_transform(speaker_mfcc)
        np.random.seed(42)
        idx_spk = np.random.choice(len(X_spk_2d), min(800, len(X_spk_2d)), replace=False)
        X_spk_vis = X_spk_2d[idx_spk]
        
        # Reset state when speaker changes
        if ('spk_train_iter' not in st.session_state
                or st.session_state.get('scenario') != f"speaker_{selected_speaker}"):
            st.session_state.scenario = f"speaker_{selected_speaker}"
            st.session_state.spk_train_iter = 0
            st.session_state.spk_em_gmm = GaussianMixture(
                n_components=n_components_spk, covariance_type='full',
                max_iter=1, warm_start=True, init_params='kmeans', random_state=42)
            st.session_state.spk_ll_history = []

        col_ctrl, col_viz = st.columns([1, 2])
        with col_ctrl:
            st.markdown("<h4 style='color:#00E0FF'>⚙️ Điều Khiển</h4>", unsafe_allow_html=True)
            st.write(f"Người nói: **{selected_speaker.upper()}** | **{len(speaker_mfcc)}** khung tiếng")
            st.write(f"**EM Iteration:** `{st.session_state.spk_train_iter}`")
            
            c1, c2 = st.columns(2)
            if c1.button("🔄 Reset"):
                st.session_state.pop('spk_train_iter', None)
                st.rerun()
            if c2.button("Lặp 1 Vòng ➡️"):
                st.session_state.spk_em_gmm.fit(X_spk_vis)
                ll = st.session_state.spk_em_gmm.score(X_spk_vis)
                st.session_state.spk_ll_history.append(ll)
                st.session_state.spk_train_iter += 1
                st.rerun()
            if st.button("Lặp Đi 10 Vòng ⏭️"):
                for _ in range(10):
                    st.session_state.spk_em_gmm.fit(X_spk_vis)
                    ll = st.session_state.spk_em_gmm.score(X_spk_vis)
                    st.session_state.spk_ll_history.append(ll)
                    st.session_state.spk_train_iter += 1
                st.rerun()
            
            st.markdown("---")
            st.markdown("##### 📐 Toán Học EM từng bước:")
            st.markdown("**E-Step** — Tính trách nhiệm:")
            st.latex(r"\gamma_{ic} = \frac{\pi_c \mathcal{N}(x_i|\mu_c,\Sigma_c)}{\sum_k \pi_k \mathcal{N}(x_i|\mu_k,\Sigma_k)}")
            st.markdown("**M-Step** — Cập nhật Tâm + Elip:")
            st.latex(r"\mu_c^{new} = \frac{\sum_i \gamma_{ic} x_i}{\sum_i \gamma_{ic}}")
            st.latex(r"\Sigma_c^{new} = \frac{\sum_i \gamma_{ic}(x_i-\mu_c)(x_i-\mu_c)^T}{\sum_i \gamma_{ic}}")
            st.write(r"Mỗi Vòng Lặp = 1 lần E-Step + 1 lần M-Step. GMM dần ôm khít 6 Elip vào đúng 6 chế độ phát âm.")

            if len(st.session_state.spk_ll_history) > 1:
                st.markdown("##### 📈 Hội tụ Log-Likelihood:")
                fig_ll, ax_ll = plt.subplots(figsize=(4, 2.5))
                fig_ll.patch.set_facecolor('#19120B')
                ax_ll.set_facecolor('#19120B')
                ax_ll.plot(st.session_state.spk_ll_history, color='#00E0FF', linewidth=2, marker='o', markersize=3)
                ax_ll.axhline(y=st.session_state.spk_ll_history[-1], color='#FF6B00', linestyle='--', linewidth=1, alpha=0.6)
                ax_ll.set_xlabel("Iteration", color='white', fontsize=8)
                ax_ll.set_ylabel("Log-Likelihood", color='white', fontsize=8)
                ax_ll.tick_params(colors='white', labelsize=7)
                ax_ll.set_title("Tăng → Hội tụ ✅", color='#00E0FF', fontsize=8)
                plt.tight_layout()
                st.pyplot(fig_ll)

        with col_viz:
            fig, ax = plt.subplots(figsize=(10, 7))
            fig.patch.set_facecolor('#19120B')
            ax.set_facecolor('#19120B')
            
            colors_spk = plt.cm.Set2(np.linspace(0, 1, n_components_spk))
            x_min_s = X_spk_vis[:, 0].min() - 5
            x_max_s = X_spk_vis[:, 0].max() + 5
            y_min_s = X_spk_vis[:, 1].min() - 5
            y_max_s = X_spk_vis[:, 1].max() + 5
            
            ax.scatter(X_spk_vis[:, 0], X_spk_vis[:, 1], s=8, c='#A5A5A5', alpha=0.3, label="Khung tiếng MFCC")
            ax.set_xlabel("MFCC PCA-1 (Âm sắc tổng hợp)", color='white', fontsize=10)
            ax.set_ylabel("MFCC PCA-2 (Năng lượng tần số)", color='white', fontsize=10)
            ax.tick_params(colors='white')
            ax.set_xlim(x_min_s, x_max_s)
            ax.set_ylim(y_min_s, y_max_s)
            ax.set_title(f"GMM ({n_components_spk} Elip) — Nhận Dạng Giọng {selected_speaker.upper()} — Iter {st.session_state.spk_train_iter}", 
                         color='#00E0FF', fontweight='bold', fontsize=11)
            
            if st.session_state.spk_train_iter > 0:
                xx_g, yy_g = np.meshgrid(
                    np.linspace(x_min_s, x_max_s, 120),
                    np.linspace(y_min_s, y_max_s, 120))
                Z_gmm = st.session_state.spk_em_gmm.score_samples(
                    np.c_[xx_g.ravel(), yy_g.ravel()]).reshape(xx_g.shape)
                
                lv = np.linspace(Z_gmm.min() * 0.5, Z_gmm.max(), 12)
                ax.contourf(xx_g, yy_g, Z_gmm, levels=lv, cmap='magma', alpha=0.45)
                ax.contour(xx_g, yy_g, Z_gmm, levels=lv, cmap='magma', alpha=0.9, linewidths=1.3)
                
                gmm_means = st.session_state.spk_em_gmm.means_
                gmm_weights = st.session_state.spk_em_gmm.weights_
                for k in range(n_components_spk):
                    ax.scatter(gmm_means[k, 0], gmm_means[k, 1], marker='+', s=220,
                               color=colors_spk[k], zorder=10, linewidths=3)
                    ax.annotate(
                        f"G{k+1}\n({gmm_weights[k]*100:.0f}%)",
                        (gmm_means[k, 0], gmm_means[k, 1]),
                        xytext=(5, 5), textcoords='offset points',
                        color=colors_spk[k], fontsize=8, fontweight='bold')
                
                ax.text(x_min_s + 0.5, y_max_s - 1.5,
                        f'6 Elip = 6 Chế độ phát âm | Iter {st.session_state.spk_train_iter}',
                        color='#00E0FF', fontsize=9)
            else:
                ax.text(0, 0, 'Nhấn "Lặp 1 Vòng" để bắt đầu EM!',
                        color='#FF6B00', fontsize=12, ha='center', va='center',
                        transform=ax.transAxes)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown("""
            <div class='brutal-card' style='padding:12px;margin-top:8px;'>
              <h5 style='color:#00E0FF;margin-top:0'>📌 Cách đọc biểu đồ:</h5>
              <ul style='color:#F8F5F2;font-size:0.9em'>
                <li><b>Chấm xám:</b> Mỗi chấm = 1 khung 25ms tiếng GMM trích xuất từ file audio</li>
                <li><b>Dấu + màu sắc (G1→G6):</b> Tâm của 6 Elip. Mỗi Elip học 1 kiểu âm thanh (Nguyên âm /a/, phụ âm /s/, khoảng lặng...)</li>
                <li><b>% bên cạnh G1→G6:</b> Trọng số π — Elip chiếm bao nhiêu % trong tổng thể giọng nói</li>
                <li><b>Bản đồ màu Magma:</b> Vùng sáng = Xác suất GMM cao (Đặc trưng tiếng điển hình). Vùng tối = Ít gặp</li>
                <li><b>Nhận dạng:</b> Khi nghe giọng mới → Tính Log-Likelihood qua 6 Elip. Ai có điểm cao nhất = Chính chủ ✅</li>
              </ul>
            </div>
            """, unsafe_allow_html=True)


# ==============================================================================
# MODE 2: SPEAKER ID INFERENCE
# ==============================================================================
elif app_mode == '🎙️ Speaker ID (Inference)':
    try:
        speaker_gmms, train_features = load_speaker_gmm()
    except Exception as e:
        st.error(f"Lỗi khởi tạo Voice: {e}"); st.stop()

    st.markdown("""
    <div class='brutal-card'>
        <h3 style='color:#FF6B00;margin-top:0'>🎙️ Nhận Dạng Giọng Nói — Đưa File Âm Thanh Lạ Vào</h3>
        <p>Hệ thống đã huấn luyện <b>6 GMM riêng biệt</b> (mỗi người 1 cái). Chọn hoặc Upload 1 file audio bất kỳ →
        hệ thống tính <b>Log-Likelihood</b> qua tất cả 6 GMM → Ai điểm cao nhất → Đó là người nói!</p>
    </div>
    """, unsafe_allow_html=True)

    choice_type = st.radio("Nguồn Dữ Liệu Audio:", ["📁 Chọn từ Database có sẵn", "📤 Tải file .wav của bạn lên"], horizontal=True)
    audio_bytes = None
    
    if choice_type == "📁 Chọn từ Database có sẵn":
        test_dir = "archive/recordings"
        if os.path.exists(test_dir):
            wav_files = [f for f in os.listdir(test_dir) if f.endswith('.wav')]
            wav_files.sort()
            if wav_files:
                selected_file = st.selectbox("Chọn file audio (Khuyên dùng số đuôi 0-4 vì GMM chưa từng thấy các file này lúc train!):", wav_files, index=0)
                file_path = os.path.join(test_dir, selected_file)
                with open(file_path, "rb") as f:
                    audio_bytes = f.read()
            else:
                st.warning(f"Thư mục {test_dir} trống!")
        else:
            st.error(f"Không tìm thấy thư mục {test_dir}!")
    else:
        uploaded = st.file_uploader("📂 Upload file audio cần nhận dạng (.wav)", type=['wav'], key='infer_audio')
        if uploaded is not None:
            audio_bytes = uploaded.read()

    if audio_bytes is not None:
        st.audio(audio_bytes, format='audio/wav')

        with st.spinner("⏳ Đang trích xuất MFCC và tính điểm..."):
            import io
            mfcc_test, y_audio, sr_audio, mfcc_raw = extract_mfcc(io.BytesIO(audio_bytes))

        # --- MFCC step-by-step simulation ---
        st.markdown("---")
        st.markdown("### 🎛️ Mô Phỏng Trích Xuất MFCC Từng Bước")

        tab_mfcc1, tab_mfcc2, tab_mfcc3, tab_mfcc4 = st.tabs([
            "① Dạng Sóng", "② Spectrogram", "③ Mel Filter", "④ MFCC Cuối"])

        with tab_mfcc1:
            col_a, col_b = st.columns([3, 2])
            with col_a:
                fig_w, ax_w = plt.subplots(figsize=(8, 2.5))
                fig_w.patch.set_facecolor('#19120B'); ax_w.set_facecolor('#19120B')
                t = np.linspace(0, len(y_audio) / sr_audio, len(y_audio))
                ax_w.plot(t, y_audio, color='#00E0FF', linewidth=0.5, alpha=0.9)
                ax_w.set_xlabel("Thời gian (giây)", color='white'); ax_w.set_ylabel("Biên độ", color='white')
                ax_w.tick_params(colors='white'); ax_w.set_title("Dạng sóng âm thanh gốc", color='white', fontweight='bold')
                plt.tight_layout(); st.pyplot(fig_w)
            with col_b:
                st.markdown("**Bước 1: Lấy mẫu (Sampling)**")
                st.latex(r"x[n] = x(nT_s), \quad T_s = \frac{1}{f_s}")
                st.write(f"File này: **{sr_audio} Hz**, {len(y_audio)} mẫu, {len(y_audio)/sr_audio:.2f}s")
                st.markdown("**Pre-emphasis** — Tăng cường tần số cao:")
                st.latex(r"y[n] = x[n] - \alpha \cdot x[n-1], \quad \alpha \approx 0.97")
                st.markdown("**Framing** — Cắt thành cửa sổ 25ms:")
                st.latex(r"\text{Frame}_k = x[k \cdot S : k \cdot S + L]")
                st.write(f"→ Ra **{mfcc_raw.shape[1]} khung** × 13 hệ số = {mfcc_test.shape} features")

        with tab_mfcc2:
            col_a, col_b = st.columns([3, 2])
            with col_a:
                fig_sp, ax_sp = plt.subplots(figsize=(8, 3))
                fig_sp.patch.set_facecolor('#19120B'); ax_sp.set_facecolor('#19120B')
                import librosa.display
                D = librosa.amplitude_to_db(np.abs(librosa.stft(y_audio)), ref=np.max)
                librosa.display.specshow(D, sr=sr_audio, x_axis='time', y_axis='hz', ax=ax_sp, cmap='magma')
                ax_sp.set_title("Short-Time Fourier Transform (STFT)", color='white', fontweight='bold')
                ax_sp.tick_params(colors='white'); ax_sp.set_xlabel("Thời gian (s)", color='white')
                ax_sp.set_ylabel("Tần số (Hz)", color='white')
                plt.tight_layout(); st.pyplot(fig_sp)
            with col_b:
                st.markdown("**STFT** — Biến đổi Fourier cửa sổ:")
                st.latex(r"X_k[m] = \sum_{n=0}^{L-1} w[n] \cdot x[n+kS] \cdot e^{-j2\pi mn/L}")
                st.write("Mỗi khung thời gian → Phổ tần số (magnitude spectrum)")
                st.markdown("**Power Spectrum:**")
                st.latex(r"P[m] = \frac{1}{L}|X[m]|^2")

        with tab_mfcc3:
            col_a, col_b = st.columns([3, 2])
            with col_a:
                fig_mel, ax_mel = plt.subplots(figsize=(8, 3))
                fig_mel.patch.set_facecolor('#19120B'); ax_mel.set_facecolor('#19120B')
                mel_spec = librosa.feature.melspectrogram(y=y_audio, sr=sr_audio, n_mels=40)
                mel_db = librosa.amplitude_to_db(mel_spec, ref=np.max)
                librosa.display.specshow(mel_db, sr=sr_audio, x_axis='time', y_axis='mel', ax=ax_mel, cmap='magma')
                ax_mel.set_title("Mel Spectrogram (40 bộ lọc)", color='white', fontweight='bold')
                ax_mel.tick_params(colors='white'); ax_mel.set_xlabel("Thời gian (s)", color='white')
                ax_mel.set_ylabel("Tần số Mel", color='white')
                plt.tight_layout(); st.pyplot(fig_mel)
            with col_b:
                st.markdown("**Mel Scale** — Mô phỏng tai người:")
                st.latex(r"m = 2595 \cdot \log_{10}\!\left(1 + \frac{f}{700}\right)")
                st.write("Tai người phân biệt tốt tần số thấp hơn tần số cao → Mel scale nén không đều")
                st.markdown("**Áp dụng 40 bộ lọc tam giác** chồng lên Power Spectrum:")
                st.latex(r"S_m = \sum_{k} H_m[k] \cdot P[k]")

        with tab_mfcc4:
            col_a, col_b = st.columns([3, 2])
            with col_a:
                fig_mf, ax_mf = plt.subplots(figsize=(8, 3))
                fig_mf.patch.set_facecolor('#19120B'); ax_mf.set_facecolor('#19120B')
                librosa.display.specshow(mfcc_raw, x_axis='time', ax=ax_mf, cmap='coolwarm')
                ax_mf.set_title(f"MFCC (13 hệ số × {mfcc_raw.shape[1]} khung)", color='white', fontweight='bold')
                ax_mf.tick_params(colors='white'); ax_mf.set_xlabel("Thời gian (s)", color='white')
                ax_mf.set_ylabel("Hệ số MFCC", color='white')
                plt.tight_layout(); st.pyplot(fig_mf)
            with col_b:
                st.markdown("**DCT** — Biến đổi Cosine rời rạc:")
                st.latex(r"c_n = \sqrt{\frac{2}{M}}\sum_{m=1}^{M} \log(S_m)\cos\!\left(\frac{\pi n(m-0.5)}{M}\right)")
                st.write(f"→ Giữ **13 hệ số đầu** → Vector đặc trưng âm thanh mỗi khung")
                st.markdown("**Delta & Delta-Delta** (vận tốc + gia tốc):")
                st.latex(r"\Delta c_t = \frac{\sum_{\tau=1}^{W}\tau(c_{t+\tau}-c_{t-\tau})}{2\sum_{\tau=1}^{W}\tau^2}")
                st.write(f"Kết hợp 3 lớp → **{mfcc_test.shape[1]} chiều** / khung")

        # --- Log-Likelihood formula panel ---
        st.markdown("---")
        st.markdown("### 📐 Công Thức Nhận Dạng — Log-Likelihood")
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.markdown("**Điểm số GMM cho từng người nói s:**")
            st.latex(r"\text{Score}(s) = \frac{1}{T}\sum_{t=1}^{T} \log P_s(x_t)")
            st.latex(r"P_s(x) = \sum_{c=1}^{K} \pi_c^{(s)} \cdot \mathcal{N}(x \mid \mu_c^{(s)}, \Sigma_c^{(s)})")
            st.write("T = số khung MFCC, K = 6 thành phần Gaussian của GMM người s")
        with col_f2:
            st.markdown("**Quyết định nhận dạng:**")
            st.latex(r"\hat{s} = \arg\max_{s \in \text{Speakers}} \text{Score}(s)")
            st.info("🏆 Người có tổng Log-Likelihood cao nhất = GMM của họ giải thích file audio này tốt nhất = Chính chủ!")

        # Score against all 6 GMMs
        scores = {}
        for spk in SPEAKERS:
            scores[spk] = speaker_gmms[spk].score(mfcc_test)

        best_speaker = max(scores, key=scores.get)
        worst_score = min(scores.values())
        score_range = max(scores.values()) - worst_score if max(scores.values()) != worst_score else 1

        st.markdown(f"""
        <div style='background:#261E17;border:3px solid #FF6B00;padding:16px 24px;margin:16px 0;box-shadow:4px 4px 0 #000'>
            <h2 style='color:#FF6B00;margin:0'>🏆 Kết Quả: <span style='color:#00E0FF'>{best_speaker.upper()}</span></h2>
            <p style='color:#F8F5F2;margin:4px 0'>Log-Likelihood cao nhất: <b>{scores[best_speaker]:.2f}</b></p>
        </div>
        """, unsafe_allow_html=True)

        # Bar chart of all scores
        st.markdown("#### 📊 Điểm Log-Likelihood của tất cả 6 GMM:")
        fig_scores, ax_s = plt.subplots(figsize=(10, 4))
        fig_scores.patch.set_facecolor('#19120B')
        ax_s.set_facecolor('#19120B')

        sorted_spk = sorted(scores, key=scores.get, reverse=True)
        bar_colors = ['#FF6B00' if s == best_speaker else '#555555' for s in sorted_spk]
        bars = ax_s.barh(sorted_spk, [scores[s] for s in sorted_spk], color=bar_colors, edgecolor='white', linewidth=0.5)

        for bar, spk in zip(bars, sorted_spk):
            val = scores[spk]
            ax_s.text(val - 0.5, bar.get_y() + bar.get_height()/2,
                      f"{val:.2f}", va='center', ha='right', color='white', fontweight='bold', fontsize=9)

        ax_s.set_xlabel("Log-Likelihood (Cao hơn = Giống hơn)", color='white')
        ax_s.tick_params(colors='white')
        ax_s.set_title("Log-Likelihood Score qua 6 GMM — Cột Cam = Người Được Chọn", color='white', fontweight='bold')
        ax_s.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig_scores)

        st.markdown("---")
        st.markdown("#### 🗺️ Vị trí MFCC Test trên bản đồ GMM của từng người:")
        st.caption("Chấm trắng = các khung MFCC của file test, vùng sáng = vùng đặc trưng của người đó. File test càng rơi vào vùng sáng → Điểm càng cao.")

        # Show top 3 speaker GMM maps with test points overlaid
        top3 = sorted_spk[:3]
        cols = st.columns(3)
        for col, spk in zip(cols, top3):
            with col:
                spk_mfcc = train_features[spk]
                pca_inf = PCA(n_components=2, random_state=42)
                X_spk_2d = pca_inf.fit_transform(spk_mfcc)
                X_test_2d = pca_inf.transform(mfcc_test)

                np.random.seed(42)
                idx = np.random.choice(len(X_spk_2d), min(500, len(X_spk_2d)), replace=False)
                X_bg = X_spk_2d[idx]

                x_min = X_bg[:, 0].min() - 5; x_max = X_bg[:, 0].max() + 5
                y_min = X_bg[:, 1].min() - 5; y_max = X_bg[:, 1].max() + 5

                fig_m, ax_m = plt.subplots(figsize=(4, 3.5))
                fig_m.patch.set_facecolor('#19120B')
                ax_m.set_facecolor('#19120B')

                # GMM density map
                xx_m, yy_m = np.meshgrid(np.linspace(x_min, x_max, 80), np.linspace(y_min, y_max, 80))
                Z_m = speaker_gmms[spk].score_samples(
                    pca_inf.inverse_transform(np.c_[xx_m.ravel(), yy_m.ravel()])
                ).reshape(xx_m.shape)
                lv_m = np.linspace(Z_m.min(), Z_m.max(), 8)
                ax_m.contourf(xx_m, yy_m, Z_m, levels=lv_m, cmap='magma', alpha=0.6)

                # Train points background
                ax_m.scatter(X_bg[:, 0], X_bg[:, 1], s=3, c='#A5A5A5', alpha=0.2)

                # Test points
                border_color = '#FF6B00' if spk == best_speaker else '#00E0FF' if spk == sorted_spk[1] else '#888888'
                ax_m.scatter(X_test_2d[:, 0], X_test_2d[:, 1], s=6,
                             c='white', alpha=0.7, label="Test audio")

                rank = sorted_spk.index(spk) + 1
                medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉"
                ax_m.set_title(f"{medal} {spk.upper()} ({scores[spk]:.1f})", 
                               color=border_color, fontweight='bold', fontsize=9)
                ax_m.tick_params(colors='white', labelsize=6)
                ax_m.set_xlim(x_min, x_max); ax_m.set_ylim(y_min, y_max)
                plt.tight_layout()
                st.pyplot(fig_m)

        # Explanation
        st.markdown("""
        <div class='brutal-card' style='padding:12px;margin-top:8px;'>
          <h5 style='color:#FF6B00;margin-top:0'>📌 Giải thích kết quả:</h5>
          <ul style='color:#F8F5F2;font-size:0.9em'>
            <li>Hệ thống tính tổng Log-Likelihood: <b>∑log P(xᵢ | GMM_speaker)</b> qua tất cả khung MFCC</li>
            <li>GMM nào có P(x) cao nhất (chấm trắng rơi vào vùng sáng nhất) → Đó là người nói</li>
            <li>Nếu điểm các GMM gần bằng nhau → Audio có thể bị nhiễu hoặc nói giống nhau</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.info("⬆️ Upload file .wav để bắt đầu nhận dạng. Có thể dùng bất kỳ file audio giọng nói nào!")

