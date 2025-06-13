# Çalıştırmak için: streamlit run .\app.py

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib, json
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from streamlit_rag import setup_rag_chain
import html
                    
is_dark = False
try:
    is_dark = st.get_option("theme.base") == "dark"
except:
    pass

# Renkler
def get_color(light, dark):
    return dark if is_dark else light

# ─────────────────────────── Sayfa Ayarları ────────────────────────────
st.set_page_config(
    page_title="Meme Kanseri Teşhis Sistemi",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sekme ve buton için CSS ekle
st.markdown('''
    <style>
    .stTabs [data-baseweb="tab"] {
        font-size: 1.25rem !important;
        height: 60px !important;
        padding-top: 16px !important;
        padding-bottom: 16px !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #1976d2 0%, #64b5f6 100%) !important;
        color: #fff !important;
        font-weight: bold !important;
        box-shadow: 0 4px 16px rgba(25, 118, 210, 0.10);
        border-bottom: 3px solid #1976d2 !important;
        transition: background 0.3s, color 0.3s;
    }
    .stButton>button {
        background: linear-gradient(90deg, #1976d2, #64b5f6);
        color: white;
        font-size: 1.1rem;
        font-weight: bold;
        padding: 0.75rem 2.5rem;
        border-radius: 8px;
        border: none;
        margin-top: 10px;
        margin-bottom: 10px;
        transition: 0.2s;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #1565c0, #1976d2);
        color: #fff;
        transform: scale(1.04);
    }
    .stCaption, [data-testid="stCaption"] {
        font-size: 1.3rem !important;
    }
    </style>
''', unsafe_allow_html=True)

# ─────────────────────────── Veri Yükleme (Raw) ─────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/data.csv")
    df = df.drop(columns=["Unnamed: 32", "id"], errors="ignore")
    return df

df = load_data()

# ─────────────────────────── Model & Artefaktlar ────────────────────────
@st.cache_resource
def load_artifacts():
    model = load_model("breast_mlp.h5", compile=False)
    scaler = joblib.load("scaler.pkl")
    threshold = json.load(open("threshold.json"))["threshold"]
    features = pd.read_csv("data/feature_order.csv", header=None)[0].tolist()
    return model, scaler, threshold, features

model, scaler, THRESHOLD, FEATURES = load_artifacts()

# ─────────────────────────── EDA Fonksiyonları ─────────────────────────
@st.cache_resource
def plot_class_distribution(df):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(8,6))
    sns.set_theme(style="whitegrid")
    sns.countplot(
        data=df,
        x="diagnosis",
        hue="diagnosis",
        palette=["#43A047", "#E53935"],
        ax=ax
    )
    legend = ax.get_legend()
    if legend:
        legend.remove()
    ax.set_title("Tümör Sınıf Dağılımı", fontsize=14, pad=20, color="#1B5E20")
    ax.set_xlabel("Teşhis (M = Malign, B = Benign)", fontsize=12, color="#424242")
    ax.set_ylabel("Sayı", fontsize=12, color="#424242")
    plt.tight_layout()
    return fig

@st.cache_resource
def plot_corr_heatmap(df):
    corr = df.drop(columns=["diagnosis"], errors="ignore").corr()
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(16,12))
    sns.set_theme(style="whitegrid")
    sns.heatmap(
        corr,
        cmap="RdYlGn",
        linewidths=0.5,
        ax=ax,
        annot=False,
        center=0
    )
    ax.set_title("Özellik Korelasyon Haritası", fontsize=14, pad=20, color="#1B5E20")
    plt.tight_layout()
    return fig

@st.cache_resource
def plot_radius_box(df):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(8,6))
    sns.set_theme(style="whitegrid")
    sns.boxplot(
        data=df,
        x="diagnosis",
        y="radius_mean",
        hue="diagnosis",
        palette=["#43A047", "#E53935"],
        ax=ax
    )
    legend = ax.get_legend()
    if legend:
        legend.remove()
    ax.set_title("Sınıflara Göre Yarıçap Dağılımı", fontsize=14, pad=20, color="#1B5E20")
    ax.set_xlabel("Teşhis", fontsize=12, color="#424242")
    ax.set_ylabel("Ortalama Yarıçap", fontsize=12, color="#424242")
    plt.tight_layout()
    return fig

@st.cache_resource
def plot_radius_hist(df):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(8,6))
    sns.set_theme(style="whitegrid")
    sns.histplot(
        data=df,
        x="radius_mean",
        hue="diagnosis",
        bins=30,
        kde=True,
        palette=["#43A047", "#E53935"],
        ax=ax
    )
    ax.set_title("Yarıçap Dağılımı (Histogram + KDE)", fontsize=14, pad=20, color="#1B5E20")
    ax.set_xlabel("Ortalama Yarıçap", fontsize=12, color="#424242")
    ax.set_ylabel("Frekans", fontsize=12, color="#424242")
    plt.tight_layout()
    return fig

@st.cache_resource
def plot_pca(df):
    features_only = df.drop(columns=["diagnosis"], errors="ignore")
    scaled = StandardScaler().fit_transform(features_only)
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(scaled)
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(8,6))
    sns.set_theme(style="whitegrid")
    sns.scatterplot(
        x=pcs[:,0],
        y=pcs[:,1],
        hue=df["diagnosis"],
        palette=["#43A047", "#E53935"],
        ax=ax
    )
    ax.set_title("PCA 2D Projeksiyonu", fontsize=14, pad=20, color="#1B5E20")
    ax.set_xlabel("PCA 1", fontsize=12, color="#424242")
    ax.set_ylabel("PCA 2", fontsize=12, color="#424242")
    plt.tight_layout()
    return fig

# ─────────────────────────── Uygulama Başlığı ───────────────────────────
st.title("🏥 Meme Kanseri Teşhis Sistemi")
st.write("""
Bu uygulama, Wisconsin Meme Kanseri veri seti üzerinde eğitilmiş derin öğrenme modeli ile tümörün iyi huylu (benign) veya kötü huylu (malign) olup olmadığını tahmin eder.
Ayrıca kullanıcılar, veri analizi, model değerlendirmesi, ve SağlıkGPT adlı etkileşimli chatbot sekmeleri aracılığıyla sistemi daha derinlemesine keşfedebilirler.
""")

sekme_adi1 = "🔍 Teşhis Tahmini"
sekme_adi2 = "🤖 SağlıkGPT Chatbot"
sekme_adi3 = "📊 Veri Analizi"
sekme_adi4 = "📈 Model Değerlendirmesi"
tab2, tab1, tab3, tab4 = st.tabs([sekme_adi1, sekme_adi2, sekme_adi3, sekme_adi4])

with tab2:
    st.header("Teşhis Tahmini")
    st.write("CSV dosyası yükleyerek veya özellikleri manuel olarak girerek tahmin alabilirsiniz. Bilgi: Yüklediğiniz CSV dosyasının sütun sırası, 'feature_order.csv' dosyasında belirtilen sırayla eşleşmelidir. Ayrıca proje klasöründe, data klasörü altında yer alan 'iyi.csv' (iyi huylu tümör örneği) ve 'kötü.csv' (kötü huylu tümör örneği) dosyalarını örnek veri olarak kullanabilirsiniz.")
    uploaded = st.file_uploader("CSV Dosyası Yükle (30 özellik)", type=["csv"])
    if uploaded:
        try:
            df_in = pd.read_csv(uploaded)
            
            # Sütun kontrolü
            if list(df_in.columns) != FEATURES:
                st.error("Hata: Sütunlar beklenen sırayla eşleşmiyor. Lütfen feature_order.csv dosyasını kontrol edin.")
                st.stop()
            
            # Eksik veri kontrolü
            if df_in.isnull().any().any():
                st.error("Hata: Veri setinde eksik değerler var. Lütfen tüm değerlerin dolu olduğundan emin olun.")
                st.stop()
            
            # Veri tipi kontrolü
            if not all(df_in.dtypes == 'float64'):
                st.error("Hata: Tüm değerler sayısal olmalıdır. Lütfen veri setinizi kontrol edin.")
                st.stop()
            
            # Değer aralığı kontrolü
            if (df_in < 0).any().any():
                st.error("Hata: Negatif değerler bulunamaz. Lütfen veri setinizi kontrol edin.")
                st.stop()
            
            X = scaler.transform(df_in[FEATURES])
            prob = model.predict(X)[0][0]
            pred = int(prob > THRESHOLD)
            
            st.subheader("Malign Olasılığı")
            st.write(f"Malign (Kötü Huylu) Olasılığı: %{prob*100:.1f}")
            st.write(f"Eşik: {THRESHOLD}")
            
            # Sonucu renkli kutuda göster
            if pred:
                st.markdown("""
                <div style='background:#ffcdd2; color:#b71c1c; padding:1rem; border-radius:8px; font-weight:bold; margin-bottom:1rem;'>
                Tahmin: Malign (Kötü Huylu)
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='background:#c8e6c9; color:#1b5e20; padding:1rem; border-radius:8px; font-weight:bold; margin-bottom:1rem;'>
                Tahmin: Benign (İyi Huylu)
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Veri yükleme hatası: {str(e)}")
            st.stop()
    else:
        st.subheader("Manuel Giriş")
        with st.form("prediction_form"):
            cols = st.columns(3)
            values = []
            for idx, feat in enumerate(FEATURES):
                with cols[idx % 3]:
                    try:
                        value = st.number_input(
                            feat,
                            value=0.0,
                            format="%.3f",
                            min_value=0.0,
                            help=f"{feat} için değer giriniz (0'dan büyük olmalı)"
                        )
                        values.append(value)
                    except Exception as e:
                        st.error(f"{feat} için geçersiz değer: {str(e)}")
                        st.stop()
            
            submit_button = st.form_submit_button("🔎 Tahmin Et")
            if submit_button:
                try:
                    # Değer kontrolü
                    if any(v < 0 for v in values):
                        st.error("Hata: Negatif değerler girilemez.")
                        st.stop()
                    
                    df_manual = pd.DataFrame([values], columns=FEATURES)
                    X = scaler.transform(df_manual[FEATURES])
                    prob = model.predict(X)[0][0]
                    pred = int(prob > THRESHOLD)
                    
                    st.subheader("Malign Olasılığı")
                    st.write(f"Malign (Kötü Huylu) Olasılığı: %{prob*100:.1f}")
                    st.write(f"Eşik: {THRESHOLD}")
                    
                    if pred:
                        st.markdown("""
                        <div style='background:#ffcdd2; color:#b71c1c; padding:1rem; border-radius:8px; font-weight:bold; margin-bottom:1rem;'>
                        Tahmin: Malign (Kötü Huylu)
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div style='background:#c8e6c9; color:#1b5e20; padding:1rem; border-radius:8px; font-weight:bold; margin-bottom:1rem;'>
                        Tahmin: Benign (İyi Huylu)
                        </div>
                        """, unsafe_allow_html=True)
                        
                except Exception as e:
                    st.error(f"Tahmin hatası: {str(e)}")
                    st.stop()

with tab1:
    st.header("SağlıkGPT Chatbot")
    
    # Chat görünümü için CSS
    st.markdown("""
        <style>
        .chat-message {
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            display: flex;
            flex-direction: row;
            align-items: flex-start;
            gap: 0.75rem;
        }
        .chat-message.user {
            background-color: #2b313e;
        }
        .chat-message.bot {
            background-color: #444654;
        }
        .chat-message .avatar {
            width: 2rem;
            height: 2rem;
            border-radius: 0.25rem;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.2rem;
        }
        .chat-message .message {
            color: #fff;
            font-size: 1rem;
            line-height: 1.5;
        }
        .stTextInput>div>div>input {
            background-color: #40414f;
            color: white;
            border: 1px solid #565869;
        }
        .stTextInput>div>div>input:focus {
            border-color: #19c37d;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Zinciri oluştur
    rag_chain = setup_rag_chain()

    # Chat geçmişini saklamak için
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Hoş geldin mesajı
    if not st.session_state.messages:
        st.session_state.messages.append({
            "role": "bot",
            "content": "👋 Merhaba, ben SağlıkGPT! Yalnızca meme kanseri hakkında güvenilir ve kaynaklara dayalı bilgiler sunabilirim; tıbbi teşhis yerine geçmem ama bu konuda aklınızdaki soruları yanıtlamaya hazırım. Ne sormak istersiniz?"
        })

    # Mesajları göster
    for message in st.session_state.messages:
        with st.container():
            st.markdown(f"""
                <div class="chat-message {message['role']}">
                    <div class="avatar">
                        {'👤' if message['role'] == 'user' else '🤖'}
                    </div>
                    <div class="message">
                        {message['content']}
                    </div>
                </div>
            """, unsafe_allow_html=True)

    # Kullanıcı girişi ve gönderme
    with st.form("chat_form", clear_on_submit=True):
        user_question = st.text_input("💬 Sorunuzu buraya yazın:", key="user_input")
        submit_button = st.form_submit_button("Gönder")
    if submit_button and user_question.strip():
        # Kullanıcı mesajını ekle
        st.session_state.messages.append({"role": "user", "content": user_question})
        # Bot yanıtını al
        with st.spinner("Yanıt oluşturuluyor..."):
            result = rag_chain.invoke({"input": user_question.strip()})
            bot_response = result["answer"]
        # Bot yanıtını ekle
        st.session_state.messages.append({"role": "bot", "content": bot_response})
        # Sayfayı yenile
        st.rerun()

    # Örnek Sorular kutusu (formun hemen altında)
    st.markdown("""
        <div style="background-color:#444654; border-radius:10px; padding:1.2rem 1.5rem 1.2rem 1.5rem; margin-bottom:1.5rem; border:1.5px solid #b0bec5;">
            <div style="font-weight:bold; font-size:1.15rem; margin-bottom:0.5rem; color:#fff;">
                📝 Örnek Sorular
            </div>
            <div style="color:#e0e0e0; font-size:1.05rem; margin-bottom:0.7rem;">
                 Aşağıdaki sorular, chatbot sisteminin cevaplayabileceği örnek sorulardır. Değerlendirme veya gösterim için bu tür soruları kullanabilir.
            </div>
            <ul style="margin-left:1.2rem; color:#fff; font-size:1.05rem;">
                <li>Meme kanseri nedir?</li>
                <li>Meme kanserinden korunma yolları nelerdir?</li>
                <li>Meme kanseri risk faktörlerinin azaltılması nasıl sağlanabilir?</li>
                <li>Meme kanserinde belirtiler nelerdir?</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

with tab3:
    st.header("Veri Analizi ve Görselleştirme")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Sınıf Dağılımı")
        st.pyplot(plot_class_distribution(df))
        st.caption("Veri setindeki benign ve malign tümör sayılarının dağılımı.")
    with col2:
        st.subheader("Yarıçap Dağılımı")
        st.pyplot(plot_radius_box(df))
        st.caption("Sınıflara göre ortalama yarıçap değerlerinin kutu grafiği.")
    st.subheader("Korelasyon Haritası")
    st.pyplot(plot_corr_heatmap(df))
    st.caption("Özellikler arasındaki korelasyonları gösteren ısı haritası.")
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Yarıçap Histogramı")
        st.pyplot(plot_radius_hist(df))
        st.caption("Tümör yarıçapı dağılımı ve sınıflara göre yoğunluk.")
    with col4:
        st.subheader("PCA Analizi")
        st.pyplot(plot_pca(df))
        st.caption("Veri setinin 2 boyutlu PCA ile görselleştirilmiş hali.")

with tab4:
    st.header("Model Değerlendirmesi")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confusion Matrix")
        st.image("results/confusion.png")
        st.caption("Derin öğrenme modelinin karışıklık matrisi görselleştirmesi.")
    with col2:
        st.subheader("F1 Score")
        st.image("results/f1score.png")
        st.caption("Derin öğrenme modelinin F1 Score değerlendirmesi.")
    
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Random Forest")
        st.image("results/rf.png")
        st.caption("Modelin değerlendirmesi")
    with col4:
        st.subheader("XGBoost")
        st.image("results/xg.png")
        st.caption("Modelin değerlendirmesi")
    
    col5, _ = st.columns(2)
    with col5:
        st.subheader("CatBoost")
        st.image("results/cat.png")
        st.caption("Modelin değerlendirmesi")

st.markdown("""
    <hr style='margin-top:2rem;margin-bottom:0.5rem;'>
    <div style='text-align: center; color: #b71c1c; font-size: 1.1rem; margin-bottom: 1rem;'>
        <b>⚠️ Uyarı:</b> Bu uygulama yalnızca eğitim ve ön bilgi amaçlıdır, tıbbi teşhis veya tedavi yerine geçmez. Lütfen kesin tanı ve tedavi için bir sağlık profesyoneline başvurun.
    </div>
    <div style='text-align: center; color: #bdbdbd; font-size: 1.1rem; margin-bottom: 2rem;'>
        Mahmut Kerem Erden - k.erden03@gmail.com
    </div>
""", unsafe_allow_html=True)
