# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder

# ---------------------------
# Config page
# ---------------------------
st.set_page_config(page_title="TTK_StatTestIA", page_icon="💙", layout="wide")
PRIMARY = "#0E6BA8"
ACCENT = "#DFF6FF"

st.markdown(f"""
    <style>
    body {{ background: linear-gradient(180deg, #ffffff, {ACCENT}); }}
    .title {{ text-align:center; font-size:46px; color:{PRIMARY}; font-weight:800; margin-top:25px; }}
    .subtitle {{ text-align:center; color:#333; font-size:20px; margin-bottom:25px; }}
    .heart {{
      display:inline-block;
      font-size:70px;
      color:{PRIMARY};
      animation: beat 1s infinite;
      transform-origin: center;
      margin-bottom: 10px;
    }}
    @keyframes beat {{
      0% {{ transform: scale(1); }}
      25% {{ transform: scale(1.12); }}
      50% {{ transform: scale(1); }}
      75% {{ transform: scale(1.12); }}
      100% {{ transform: scale(1); }}
    }}
    .big-btn {{
      background-color:{PRIMARY};
      color:white;
      padding:18px 40px;
      border-radius:12px;
      font-weight:600;
      text-decoration:none;
      border:none;
      font-size:18px;
      transition: 0.3s ease;
    }}
    .big-btn:hover {{ background-color:#0b5f91; transform:scale(1.08); }}
    .button-container {{
      display:flex; justify-content:center; gap:40px; margin-top:35px;
    }}
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# Load model
# ---------------------------
MODEL_FILE = "model.pkl"
try:
    model_payload = joblib.load(MODEL_FILE)
    model = model_payload["model"]
    encoders = model_payload.get("encoders", {})
    FEATURES = model_payload["features"]
    STATS = model_payload["stats"]
    model_loaded = True
except FileNotFoundError:
    model_loaded = False
    model = None
    encoders = {}
    FEATURES = []
    STATS = {}

# ---------------------------
# Sidebar navigation
# ---------------------------
st.sidebar.title("💙 TTK_StatTestIA")
menu = st.sidebar.radio("Navigation", ["🏠 Accueil", "📂 Importer", "📊 Visualisations", "🧪 Tests", "🔮 Prédiction (IA)"])

# ---------------------------
# ACCUEIL
# ---------------------------
if menu == "🏠 Accueil":
    st.markdown("<div style='text-align:center'>", unsafe_allow_html=True)
    st.markdown("<div class='heart'>💙</div>", unsafe_allow_html=True)
    st.markdown("<div class='title'>TTK_StatTestIA</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Explorez vos données, testez vos hypothèses et prédisez le risque de diabète grâce à une IA légère et intuitive.</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="button-container">
        <button class="big-btn">📂 Importer des données</button>
        <button class="big-btn">📊 Visualisations</button>
        <button class="big-btn">🧪 Tests</button>
        <button class="big-btn">🔮 Prédiction IA</button>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='text-align:center; margin-top:50px; color:#444;'>Bienvenue sur TTK_StatTestIA, votre plateforme d’analyse statistique interactive et de prédiction assistée par intelligence artificielle.</div>", unsafe_allow_html=True)

# ---------------------------
# IMPORTATION
# ---------------------------
elif menu == "📂 Importer":
    st.header("📂 Importation des données")
    uploaded_file = st.file_uploader("Choisissez un fichier CSV/Excel (ex : BD_DIABETE.csv)", type=["csv", "xlsx"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
        st.session_state["df"] = df
        st.success("✅ Importation réussie.")
        st.dataframe(df.head())

# ---------------------------
# VISUALISATIONS
# ---------------------------
elif menu == "📊 Visualisations":
    st.header("📊 Visualisation interactive des données")
    df = st.session_state.get("df", None)

    if df is None:
        st.warning("⚠️ Importez d'abord un fichier de données.")
    else:
        xcol = st.selectbox("Variable X", df.columns)
        ycol = st.selectbox("Variable Y (optionnel)", ["Aucune"] + list(df.columns))
        chart_type = st.selectbox("Type de graphique", ["Histogramme", "Boîte à moustaches", "Nuage de points", "Diagramme circulaire"])
        color_choice = st.color_picker("🎨 Choisissez une couleur", value="#0E6BA8")

        # Création des graphiques
        if chart_type == "Histogramme":
            fig = px.histogram(df, x=xcol, nbins=20, color_discrete_sequence=[color_choice], title=f"Histogramme de {xcol}")
            interpretation = f"On observe la répartition de la variable **{xcol}**. Si une barre domine, cela signifie que cette valeur est plus fréquente dans notre échantillon."

        elif chart_type == "Boîte à moustaches":
            fig = px.box(df, y=xcol, color_discrete_sequence=[color_choice], title=f"Boîte à moustaches : {xcol}")
            q1, q3 = df[xcol].quantile([0.25, 0.75])
            iqr = q3 - q1
            interpretation = f"La médiane de **{xcol}** se situe autour de {df[xcol].median():.2f}. Les valeurs comprises entre {q1:.2f} et {q3:.2f} représentent la majorité des données. Des points éloignés pourraient indiquer des valeurs extrêmes."

        elif chart_type == "Nuage de points":
            if ycol == "Aucune":
                st.warning("Veuillez choisir une variable Y pour un nuage de points.")
                st.stop()
            fig = px.scatter(df, x=xcol, y=ycol, color_discrete_sequence=[color_choice], title=f"Nuage de points : {xcol} vs {ycol}")
            correlation = df[xcol].corr(df[ycol])
            if correlation > 0.5:
                interpretation = f"Plus {xcol} augmente, plus {ycol} tend à augmenter. On observe une corrélation positive entre ces deux variables."
            elif correlation < -0.5:
                interpretation = f"Lorsque {xcol} augmente, {ycol} diminue. Il existe une corrélation négative marquée entre les deux variables."
            else:
                interpretation = f"Aucune relation claire n’apparaît entre **{xcol}** et **{ycol}**, les points semblent dispersés sans tendance nette."

        else:
            counts = df[xcol].value_counts().reset_index()
            counts.columns = [xcol, "Valeurs"]
            fig = px.pie(counts, names=xcol, values="Valeurs", color_discrete_sequence=px.colors.qualitative.Pastel, title=f"Répartition de {xcol}")
            top = counts.iloc[0][xcol]
            val = counts.iloc[0]["Valeurs"]
            interpretation = f"La modalité **{top}** est la plus représentée ({val} observations). Cela indique que cette catégorie est dominante dans l’échantillon."

        st.plotly_chart(fig, use_container_width=True)
        st.info(f"🧠 Interprétation : {interpretation}")

# ---------------------------
# TESTS NON PARAMÉTRIQUES
# ---------------------------
elif menu == "🧪 Tests":
    st.header("🧪 Tests non paramétriques")
    df = st.session_state.get("df", None)

    if df is None:
        st.warning("Importez d'abord des données.")
    else:
        num_cols = df.select_dtypes(include=np.number).columns
        if len(num_cols) < 2:
            st.warning("Besoin d'au moins deux colonnes numériques.")
        else:
            col1, col2 = st.columns(2)
            var1 = col1.selectbox("Variable A", num_cols)
            var2 = col2.selectbox("Variable B", num_cols)
            test = st.selectbox("Choisir un test", ["Mann-Whitney", "Wilcoxon", "Kruskal-Wallis", "Spearman"])

            if st.button("Exécuter le test"):
                try:
                    if test == "Mann-Whitney":
                        stat, p = stats.mannwhitneyu(df[var1], df[var2])
                        interpretation = "Le test compare les médianes de deux groupes indépendants."
                    elif test == "Wilcoxon":
                        stat, p = stats.wilcoxon(df[var1], df[var2])
                        interpretation = "Ce test vérifie si deux mesures appariées diffèrent significativement."
                    elif test == "Kruskal-Wallis":
                        stat, p = stats.kruskal(df[var1], df[var2])
                        interpretation = "Ce test compare la distribution de plusieurs groupes pour voir s’ils diffèrent."
                    else:
                        stat, p = stats.spearmanr(df[var1], df[var2])
                        interpretation = "Il mesure la force de la relation monotone entre deux variables numériques."

                    st.success(f"Statistique = {stat:.3f} | p-valeur = {p:.4f}")
                    if p < 0.05:
                        st.info(f"✅ Résultat significatif : {interpretation}. Il existe une différence ou une relation notable.")
                    else:
                        st.warning(f"⚠️ Pas de différence significative détectée selon {test}.")
                except Exception as e:
                    st.error(f"Erreur : {e}")

# ---------------------------
# PREDICTION (IA)
# ---------------------------
elif menu == "🔮 Prédiction (IA)":
    st.header("🔮 Prédiction du risque de diabète")

    if not model_loaded:
        st.error(f"Le fichier {MODEL_FILE} est introuvable. Exécutez d'abord train_model.py pour générer le modèle.")
        st.stop()

    st.write("### Entrez les valeurs pour prédire le risque :")
    user_input = {}
    for feat in FEATURES:
        info = STATS.get(feat, {})
        if info.get("type") == "categorical":
            user_input[feat] = st.selectbox(feat, info.get("classes", []))
        else:
            minv, maxv, meanv = info.get("min", 0), info.get("max", 1), info.get("mean", 0.5)
            user_input[feat] = st.number_input(feat, minv, maxv, meanv)

    if st.button("🔮 Prédire"):
        try:
            input_df = pd.DataFrame([user_input])
            for col, le in encoders.items():
                if col in input_df.columns:
                    input_df[col] = le.transform(input_df[col].astype(str))
            prob = model.predict_proba(input_df)[0][1] * 100

            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=prob, number={'suffix': "%"},
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': PRIMARY},
                       'steps': [{'range': [0, 30], 'color': "#BEE6FF"},
                                 {'range': [30, 70], 'color': "#FDDC6B"},
                                 {'range': [70, 100], 'color': "#FF9A9A"}]},
                title={'text': "Probabilité de diabète"}
            ))
            st.plotly_chart(fig, use_container_width=True)

            if prob < 30:
                st.success(f"Faible risque ({prob:.1f}%)")
            elif prob < 70:
                st.warning(f"Risque modéré ({prob:.1f}%)")
            else:
                st.error(f"Risque élevé ({prob:.1f}%)")

        except Exception as e:
            st.error(f"Erreur : {e}")
