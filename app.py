import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss, roc_curve, confusion_matrix)
import openpyxl

# ==========================================
# CONSTANTS & CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Laboratorium Regresji Logistycznej",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# SIDEBAR - CONTROL PANEL
# ==========================================
st.sidebar.title("🛠️ Panel Kontrolny")
st.sidebar.markdown("---")

# 1. Dataset Selection
st.sidebar.subheader("1. Wybór Zbioru Danych")
dataset_option = st.sidebar.selectbox(
    "Wybierz dane do analizy:",
    [
        "Wiek vs Zgon (data.xlsx)",
        "Breast Cancer (scikit-learn)",
        "Syntetyczny zbalansowany",
        "Syntetyczny niezbalansowany (90/10)",
        "Wgraj własny plik"
    ]
)

# Kontrolka widoczna TYLKO wtedy, gdy użytkownik chce przetestować zupełnie inny, własny plik
if dataset_option == "Wgraj własny plik":
    uploaded_file = st.sidebar.file_uploader("Wybierz plik CSV lub XLSX", type=["csv", "xlsx"])

# 2. Model Hyperparameters
st.sidebar.markdown("---")
st.sidebar.subheader("2. Parametry Modelu")
train_size = st.sidebar.slider("Podział Train/Test (% zbioru treningowego)", 50, 90, 75) / 100
use_scaler = st.sidebar.checkbox("Użyj standaryzacji (StandardScaler)", value=True)
class_weight = st.sidebar.selectbox("Wagi klas (class_weight)", ["None", "balanced"])

# 3. Calibration Settings
st.sidebar.markdown("---")
st.sidebar.subheader("3. Kalibracja Prawdopodobieństwa")
calibration_method = st.sidebar.selectbox(
    "Metoda kalibracji:",
    ["Brak (Model bazowy)", "Metoda Platta (Sigmoid)", "Regresja Izotoniczna"]
)

# ==========================================
# MAIN PAGE STRUCTURE
# ==========================================
st.title("📊 Interaktywne Środowisko Regresji Logistycznej")
st.markdown(
    "Aplikacja wspomagająca zrozumienie prawdopodobieństwa, szansy oraz kalibracji modeli klasyfikacji binarnej.")

# Definiowanie głównych sekcji w menu poziomym (Tabs)
tab1, tab2, tab3 = st.tabs([
    "📘 Zadanie 1: Kompendium Wiedzy i Teoria",
    "🟪 Zadanie 2: Laboratorium Modelowania",
    "📗 Zadanie 3: Arkusz Ćwiczeń i Podsumowanie"
])

# ==========================================
# TAB 1: ZADANIE 1 - TEORIA
# ==========================================
with tab1:
    st.header("📘 Zadanie 1: Prawdopodobieństwo, Szansa (Odds) i Logit")
    st.markdown("---")

    # Podsekcja 1: Wyjaśnienie pojęć
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🎲 Prawdopodobieństwo (p)")
        st.info(
            "Stosunek liczby przypadków sprzyjających zdarzeniu do liczby wszystkich możliwych przypadków. "
            "Przyjmuje wartości z przedziału **[0, 1]**."
        )
        st.latex(r"p = \frac{\text{sukcesy}}{\text{wszystkie próby}}")

    with col2:
        st.subheader("📈 Szansa (Odds)")
        st.success(
            "Stosunek liczby przypadków sprzyjających zdarzeniu do liczby przypadków, które mu nie sprzyjają. "
            "Przyjmuje wartości z przedziału **[0, +∞)**."
        )
        st.latex(r"\text{Odds} = \frac{p}{1-p}")

    with col3:
        st.subheader("🧬 Logit (Log-Odds)")
        st.warning(
            "Zlogarytmowany iloraz szans. Przekształca ograniczone prawdopodobieństwo na przestrzeń "
            "liczb rzeczywistych **(-∞, +∞)**, co umożliwia stosowanie regresji liniowej."
        )
        st.latex(r"\text{Logit}(p) = \ln\left(\frac{p}{1-p}\right)")

    st.markdown("---")

    # Podsekcja 2: Interaktywny kalkulator matematyczny
    st.subheader("🧮 Szybki Kalkulator Pojęć")
    p_input = st.slider(
        "Przesuń suwak, aby zobaczyć jak zmienia się Szansa i Logit w zależności od prawdopodobieństwa (p):", 0.01,
        0.99, 0.50, step=0.01)

    calc_odds = p_input / (1 - p_input)
    calc_logit = np.log(calc_odds)

    cc1, cc2, cc3 = st.columns(3)
    cc1.metric("Prawdopodobieństwo (p)", f"{p_input:.2f}")
    cc2.metric("Szansa (Odds)", f"{calc_odds:.4f}")
    cc3.metric("Logit (Log-Odds)", f"{calc_logit:.4f}")

    st.markdown("---")

    # Podsekcja 3: Interaktywny Symulator Sigmoidy
    st.subheader("📉 Interaktywny Symulator Funkcji Sigmoidalnej")
    st.markdown(
        "Regresja logistyczna szacuje prawdopodobieństwo za pomocą funkcji sigmoidalnej. "
        "Zmień parametry modelu, aby zobaczyć jak wpływają na kształt krzywej predykcji:"
    )

    sim_col1, sim_col2 = st.columns([1, 2])

    with sim_col1:
        st.markdown("**Parametry równania:**")
        st.latex(r"p(x) = \frac{1}{1 + e^{-(\theta_0 + \theta_1 \cdot x)}}")
        theta_0 = st.slider("Intercept (θ₀) - przesunięcie wykresu", -10.0, 10.0, 0.0, step=0.5)
        theta_1 = st.slider("Nachylenie (θ₁) - wpływ zmiennej X", -2.0, 2.0, 0.5, step=0.1)

        st.markdown("**Interpretacja bieżących ustawień:**")
        if theta_1 > 0:
            st.write(
                f"🟢 Każdy wzrost zmiennej X o jednostkę **zwiększa** szansę zdarzenia (Odds Ratio = {np.exp(theta_1):.2f}).")
        elif theta_1 < 0:
            st.write(
                f"🔴 Każdy wzrost zmiennej X o jednostkę **zmniejsza** szansę zdarzenia (Odds Ratio = {np.exp(theta_1):.2f}).")
        else:
            st.write("⚪ Zmienna X nie ma wpływu na prawdopodobieństwo zdarzenia.")

    with sim_col2:
        # Generowanie wykresu sigmoidy
        x_vals = np.linspace(-50, 50, 400)
        z = theta_0 + theta_1 * x_vals
        y_vals = 1 / (1 + np.exp(-z))

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(x_vals, y_vals, color="#1f77b4", linewidth=2.5, label="Funkcja sigmoidalna p(x)")
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Zmienna objaśniająca (X)")
        ax.set_ylabel("Prawdopodobieństwo (p)")
        ax.set_title(f"Krzywa Logistyczna dla θ₀={theta_0}, θ₁={theta_1}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)

# ==========================================
# TAB 2: ZADANIE 2 - LABORATORIUM MODELOWANIA
# ==========================================
with tab2:
    st.header("🟪 Zadanie 2: Budowa, Kalibracja i Ocena Modelu")
    st.markdown("---")

    # 1. PRZYGOTOWANIE I GENEROWANIE WYBRANEGO ZBIORU DANYCH
    X, y = None, None
    df_full = None
    dataset_ready = False

    if dataset_option == "Breast Cancer (scikit-learn)":
        X_df, y_df = load_breast_cancer(return_X_y=True, as_frame=True)
        X, y = X_df, y_df
        df_full = pd.concat([X, y.rename("target")], axis=1)
        dataset_ready = True

    elif dataset_option == "Syntetyczny zbalansowany":
        X_raw, y_raw = make_classification(n_samples=600, n_features=10, weights=[0.5, 0.5], random_state=42)
        X = pd.DataFrame(X_raw, columns=[f"cecha_{i}" for i in range(10)])
        y = pd.Series(y_raw, name="target")
        df_full = pd.concat([X, y], axis=1)
        dataset_ready = True

    elif dataset_option == "Syntetyczny niezbalansowany (90/10)":
        X_raw, y_raw = make_classification(n_samples=600, n_features=10, weights=[0.9, 0.1], random_state=42)
        X = pd.DataFrame(X_raw, columns=[f"cecha_{i}" for i in range(10)])
        y = pd.Series(y_raw, name="target")
        df_full = pd.concat([X, y], axis=1)
        dataset_ready = True

    elif dataset_option == "Wiek vs Zgon (data.xlsx)":
        import os

        filename = "data.xlsx"

        # Bezpośrednie wczytanie stałego pliku z katalogu obok app.py
        if os.path.exists(filename):
            try:
                df_full = pd.read_excel(filename)
                if 'age' in df_full.columns and 'target' in df_full.columns:
                    X = df_full[['age']]
                    y = df_full['target']
                    dataset_ready = True
                else:
                    st.error(f"❌ Znaleziono plik '{filename}', ale brakuje w nim kolumn 'age' lub 'target'!")
            except Exception as e:
                st.error(f"❌ Błąd podczas odczytu pliku '{filename}': {e}")
        else:
            st.error(f"❌ Krytyczny błąd: Nie znaleziono pliku '{filename}' w folderze z aplikacją! Upewnij się, że leży obok app.py.")

    elif dataset_option == "Wgraj własny plik":
        if 'uploaded_file' in locals() and uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df_full = pd.read_csv(uploaded_file)
                else:
                    df_full = pd.read_excel(uploaded_file)
                X = df_full.iloc[:, :-1]
                y = df_full.iloc[:, -1]
                df_full = df_full.rename(columns={df_full.columns[-1]: "target"})
                dataset_ready = True
            except Exception as e:
                st.error(f"❌ Błąd ładowania: {e}")
        else:
            st.info("💡 Proszę wgrać plik w panelu bocznym.")

    if dataset_ready and df_full is not None:
        # --- NOWA PODSEKCJA: INFORMACJE O DATASECIE ---
        st.subheader("📋 Podsumowanie i Struktura Zbioru Danych")

        # Metryki ogólne o zbiorze
        inf_col1, inf_col2, inf_col3, inf_col4 = st.columns(4)
        inf_col1.metric("Liczba obserwacji (wierszy)", f"{df_full.shape[0]}")
        inf_col2.metric("Liczba wszystkich kolumn", f"{df_full.shape[1]}")

        # Obliczanie rozkładu klas w zmiennej celowej (target)
        class_counts = y.value_counts()
        class_0_count = class_counts.get(0, 0)
        class_1_count = class_counts.get(1, 0)

        inf_col3.metric("Klasa 0 (Przeżycie / Negatywny)", f"{class_0_count} ({class_0_count / len(y) * 100:.1f}%)")
        inf_col4.metric("Klasa 1 (Zgon / Pozytywny)", f"{class_1_count} ({class_1_count / len(y) * 100:.1f}%)")

        # Wyświetlanie listy kolumn i pierwszych 25 wierszy
        with st.expander("🔍 Zobacz szczegóły: Lista kolumn oraz pierwsze 25 wierszy danych", expanded=True):
            st.markdown(f"**Dostępne kolumny wejściowe (cechy):** `{list(X.columns)}`")
            st.markdown("**Kolumna wynikowa (zmienna docelowa):** `target`")
            st.markdown("**Pierwsze 25 wierszy z wybranego zbioru danych:**")
            st.dataframe(df_full.head(25), use_container_width=True)

        st.markdown("---")

        # --- DALSZA CZĘŚĆ PROCESU MODELOWANIA ---
        # Podział na zbiór treningowy i testowy
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=(1 - train_size), stratify=y, random_state=42
        )

        # Budowa Pipeline'u
        steps = []
        if use_scaler:
            steps.append(("sc", StandardScaler()))

        cw_param = None if class_weight == "None" else "balanced"
        steps.append(("lr", LogisticRegression(max_iter=500, class_weight=cw_param, solver="lbfgs")))

        base_pipeline = Pipeline(steps)
        base_pipeline.fit(X_train, y_train)

        # Generowanie prawdopodobieństw modelu bazowego
        p_base = base_pipeline.predict_proba(X_test)[:, 1]

        # Modele skalibrowane (Platt i Izotoniczna)
        platt_model = CalibratedClassifierCV(base_pipeline, method="sigmoid", cv=5).fit(X_train, y_train)
        iso_model = CalibratedClassifierCV(base_pipeline, method="isotonic", cv=5).fit(X_train, y_train)

        p_platt = platt_model.predict_proba(X_test)[:, 1]
        p_iso = iso_model.predict_proba(X_test)[:, 1]

        # Wybór aktywnego prawdopodobieństwa na podstawie panelu bocznego
        if calibration_method == "Brak (Model bazowy)":
            p_active = p_base
            active_label = "Bazowy"
        elif calibration_method == "Metoda Platta (Sigmoid)":
            p_active = p_platt
            active_label = "Platt"
        else:
            p_active = p_iso
            active_label = "Isotonic"

        # --- PODSEKCJA 1: PRAWDOPODOBIEŃSTWO VS SZANSA VS LOGIT ---
        st.subheader("🔢 1. Prawdopodobieństwo, Szansa (Odds) i Logit dla próbek testowych")
        st.markdown("Porównanie transformacji wartości dla pierwszych 5 obserwacji ze zbioru testowego:")


        def odds(p):
            return p / (1 - p)


        def logit(p):
            return np.log(odds(p))


        # Przygotowanie bezpiecznych wartości (unikamy log(0) lub podziału przez 0)
        p_safe = np.clip(p_active[:5], 1e-5, 1 - 1e-5)

        df_transform = pd.DataFrame({
            "Indeks próbki": X_test.index[:5],
            "Prawdopodobieństwo (p)": p_active[:5],
            "Szansa (Odds)": odds(p_safe),
            "Logit": logit(p_safe)
        }).set_index("Indeks próbki")

        st.dataframe(df_transform.style.format("{:.4f}"))

        # --- PODSEKCJA 2: OCENA JAKOŚCI MODELU ---
        st.markdown("---")
        st.subheader("📈 2. Metryki Jakości i Kalibracji")

        m_col1, m_col2, m_col3 = st.columns(3)

        auc_roc = roc_auc_score(y_test, p_active)
        auc_pr = average_precision_score(y_test, p_active)
        brier = brier_score_loss(y_test, p_active)

        # Porównanie z bazowym w celu wyświetlenia delty (tylko jeśli wybrano kalibrację)
        brier_base = brier_score_loss(y_test, p_base)
        brier_delta = brier - brier_base if calibration_method != "Brak (Model bazowy)" else None

        m_col1.metric("AUC-ROC (Zdolność dyskryminacji)", f"{auc_roc:.3f}",
                      help="Wysoka wartość oznacza świetne rozróżnianie klas (0 i 1). Idealnie: 1.0")
        m_col2.metric("AUC-PR (Precyzja-Czułość)", f"{auc_pr:.3f}",
                      help="Kluczowa miara przy zbiorach niezbalansowanych. Idealnie: 1.0")
        m_col3.metric("Brier Score (Błąd kalibracji)", f"{brier:.3f}",
                      delta=f"{brier_delta:.3f}" if brier_delta is not None else None, delta_color="inverse",
                      help="Mierzy odchylenie prawdopodobieństwa od rzeczywistości. Im BLIŻEJ ZERA, tym lepsza kalibracja.")

        # --- PODSEKCJA 3: INTERAKTYWNE WYKRESY DIAGNOSTYCZNE ---
        st.markdown("---")
        st.subheader("📊 3. Diagnostyka Wizualna Modelu")

        fig_col1, fig_col2 = st.columns(2)

        with fig_col1:
            st.markdown("**Krzywa Kalibracji (Reliability Plot)**")
            fig_cal, ax_cal = plt.subplots(figsize=(6, 4.5))

            for name, p_v in [("Bazowy", p_base), ("Platt", p_platt), ("Isotonic", p_iso)]:
                frac_pos, mean_pred = calibration_curve(y_test, p_v, n_bins=5, strategy='quantile')
                lw_val = 3 if name == active_label else 1.5
                ax_cal.plot(mean_pred, frac_pos, marker='o', label=name, linewidth=lw_val)

            ax_cal.plot([0, 1], [0, 1], 'k--', alpha=0.5, label="Idealna kalibracja")
            ax_cal.set_xlabel("Prognozowane prawdopodobieństwo")
            ax_cal.set_ylabel("Rzeczywista częstość klasy 1")
            ax_cal.legend()
            ax_cal.grid(True, alpha=0.3)
            st.pyplot(fig_cal)

        with fig_col2:
            st.markdown("**Dobór Progu Decyzyjnego za pomocą Wskaźnika Youdena**")
            fpr, tpr, thresholds = roc_curve(y_test, p_active)

            # Wskaźnik Youdena = TPR - FPR
            youden_idx = np.argmax(tpr - fpr)
            best_threshold = thresholds[youden_idx]

            fig_roc, ax_roc = plt.subplots(figsize=(6, 4.5))
            ax_roc.plot(fpr, tpr, color='#9467bd', linewidth=2, label=f'Krzywa ROC ({active_label})')
            ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            # Zaznaczenie punktu Youdena
            ax_roc.scatter(fpr[youden_idx], tpr[youden_idx], color='red', s=100, zorder=5,
                           label=f'Próg Youdena: {best_threshold:.3f}')

            ax_roc.set_xlabel("FPR (False Positive Rate / 1 - Specyficzność)")
            ax_roc.set_ylabel("TPR (True Positive Rate / Czułość)")
            ax_roc.legend()
            ax_roc.grid(True, alpha=0.3)
            st.pyplot(fig_roc)

        # --- PODSEKCJA 4: MATIERZ POMYŁEK DLA OPTYMALNEGO PROGU ---
        st.markdown("---")
        st.subheader("🎯 4. Optymalny próg decyzyjny i Macierz Pomyłek")

        y_pred_custom = (p_active >= best_threshold).astype(int)
        cm = confusion_matrix(y_test, y_pred_custom)

        cm_col1, cm_col2 = st.columns([1, 2])

        with cm_col1:
            st.write(f"Wyznaczony optymalny próg klasyfikacji: **{best_threshold:.4f}**")
            st.write(f"• Czułość (TPR) dla progu: **{tpr[youden_idx]:.3f}**")
            st.write(f"• Skala fałszywych alarmów (FPR): **{fpr[youden_idx]:.3f}**")

            # Analiza błędów typu I i II
            błąd_typu_1 = cm[0, 1]  # FP
            błąd_typu_2 = cm[1, 0]  # FN

            st.markdown("**Analiza popełnianych błędów:**")
            st.write(f"• Błędy typu I (Fałszywie dodatnie): **{błąd_typu_1}**")
            st.write(f"• Błędy typu II (Fałszywie ujemne): **{błąd_typu_2}**")

            if błąd_typu_1 > błąd_typu_2:
                st.warning("⚠️ Model częściej popełnia **błędy typu I** (ogłasza fałszywy alarm).")
            elif błąd_typu_2 > błąd_typu_1:
                st.warning("⚠️ Model częściej popełnia **błędy typu II** (przeocza rzeczywiste zdarzenie/chorobę).")
            else:
                st.info("⚖️ Model generuje równą liczbę błędów typu I oraz typu II.")

        with cm_col2:
            # Wizualizacja tabeli macierzy pomyłek
            df_cm = pd.DataFrame(
                cm,
                index=["Rzeczywiste 0 (Zdrowy/Przeżycie)", "Rzeczywiste 1 (Chory/Zgon)"],
                columns=["Prognoza 0", "Prognoza 1"]
            )
            st.markdown("**Macierz Pomyłek:**")
            st.dataframe(df_cm.style.background_gradient(cmap='Purples', axis=None))

        # --- PODSEKCJA 5: INTERPRETACJA WSPÓŁCZYNNIKÓW ---
        st.markdown("---")
        st.subheader("🧬 5. Istotność cech i Ilorazy Szans (Odds Ratio)")

        # Pobieramy współczynniki z regresji logistycznej
        lr_model = base_pipeline.named_steps["lr"]
        coefs = lr_model.coef_.ravel()

        # Przygotowanie nazw cech
        feature_names = X.columns

        df_coefs = pd.DataFrame({
            "Cecha": feature_names,
            "Współczynnik Beta (β)": coefs,
            "Iloraz Szans (Odds Ratio = e^β)": np.exp(coefs)
        }).sort_values(by="Iloraz Szans (Odds Ratio = e^β)", ascending=False)

        st.markdown(
            "Interpretacja ilorazu szans (OR): Wartość **OR > 1** oznacza, że wzrost danej cechy zwiększa "
            "szansę na wystąpienie zdarzenia (klasa 1). Wartość **OR < 1** działa ochronnie/zmniejsza szansę."
        )

        coef_col1, coef_col2 = st.columns([2, 1])

        with coef_col1:
            st.dataframe(df_coefs.style.format({
                "Współczynnik Beta (β)": "{:.4f}",
                "Iloraz Szans (Odds Ratio = e^β)": "{:.4f}"
            }).background_gradient(subset=["Współczynnik Beta (β)"], cmap='RdYlGn'))

        with coef_col2:
            st.markdown("**Top 5 najważniejszych cech (w ujęciu OR):**")
            top_5 = df_coefs.head(5)
            for idx, row in top_5.iterrows():
                st.write(
                    f"🔹 **{row['Cecha']}**: zwiększa szansę **{row['Iloraz Szans (Odds Ratio = e^β)']:.2f}-krotnie**")

# ==========================================
# TAB 3: ZADANIE 3 - ARKUSZ ĆWICZEŃ I WNIOSKI
# ==========================================
with tab3:
    st.header("📗 Zadanie 3: Oficjalny Arkusz Ćwiczeń z Odpowiedziami")
    st.markdown(
        "Poniższy panel zawiera pełne odpowiedzi na pytania kontrolne oraz dynamicznie generowane rozwiązania zadań praktycznych na podstawie aktywnego modelu.")
    st.markdown("---")

    # ==========================================
    # CZĘŚĆ I: PYTANIA KONTROLNE
    # ==========================================
    st.subheader("📝 Część I - Pytania kontrolne")

    with st.expander("1. Na czym polega podstawowa różnica między regresją liniową a logistyczną?", expanded=False):
        st.markdown(
            "**Odpowiedź:** \n"
            "Regresja liniowa służy do przewidywania **zmiennych ciągłych** (np. cena domu, zarobki) i może przyjmować dowolne wartości od $-\infty$ do $+\infty$.  \n"
            "Regresja logistyczna służy do oceny wpływu czynników na **prawdopodobieństwo wystąpienia zdarzenia binarnego** (0 lub 1, np. przeżycie/zgon), "
            "a jej wartości wyjściowe są ograniczone przez funkcję sigmoidalną do przedziału **[0, 1]**."
        )

    with st.expander("2. Co oznacza wartość funkcji sigmoidalnej w regresji logistycznej?", expanded=False):
        st.markdown(
            "**Odpowiedź:** \n"
            "Wartość funkcji sigmoidalnej oznacza **prognozowane prawdopodobieństwo** wystąpienia analizowanego zdarzenia (czyli przynależności do klasy 1, np. zgonu) "
            "przy danych wartościach zmiennych objaśniających."
        )

    with st.expander("3. Jak interpretujemy współczynniki θ₀ (intercept) oraz θ₁ (nachylenie)?", expanded=False):
        st.markdown(
            "**Odpowiedź:** \n"
            "* $\theta_0$ (**Intercept**): Ustala punkt przecięcia osi — jest to wartość logitu (log-ilorazu szans), gdy wszystkie zmienne objaśniające (np. wiek) są równe 0.  \n"
            "* $\theta_1$ (**Nachylenie / Slope**): Oznacza przyrost log-ilorazu szans (log-odds) w przypadku wzrostu zmiennej objaśniającej o jedną jednostkę (np. o 1 rok). "
            "Jeśli $e^{\theta_1} > 1$, wzrost zmiennej zwiększa szansę na zdarzenie; jeśli $e^{\theta_1} < 1$, zmniejsza ją."
        )

    with st.expander("4. Co przedstawia logit i jaka jest jego zależność z prawdopodobieństwem zdarzenia?",
                     expanded=False):
        st.markdown(
            "**Odpowiedź:** \n"
            "Logit przedstawia **logarytm naturalny z ilorazu szans (log-odds)**. Zależność z prawdopodobieństwem $p$ wyraża się wzorem:  \n"
            "$$\\text{Logit}(p) = \\ln\\left(\\frac{p}{1-p}\\right)$$ \n"
            "Gdy prawdopodobieństwo rośnie od 0 do 1, logit rośnie nieliniowo w przedziale od $-\infty$ do $+\infty$, tworząc liniową relację ze zmiennymi objaśniającymi."
        )

    with st.expander("5. Jakie znaczenie ma iloraz szans (odds ratio) i jak go obliczamy z wyniku modelu?",
                     expanded=False):
        st.markdown(
            "**Odpowiedź:** \n"
            "Iloraz szans (**Odds Ratio - OR**) informuje, ile razy wzrośnie (lub spadnie) szansa wystąpienia zdarzenia, gdy zmienna objaśniająca wzrośnie o jednostkę. "
            "Obliczamy go poprzez podniesienie liczby $e$ (podstawy logarytmu naturalnego) do potęgi danego współczynnika:  \n"
            "$$\\text{OR} = e^{\\theta_1}$$"
        )

    with st.expander("6. Czym jest miara AUC-ROC i co informuje o jakości klasyfikacji?", expanded=False):
        st.markdown(
            "**Odpowiedź:** \n"
            "**AUC-ROC** (Area Under the ROC Curve) to miara zdolności dyskryminacyjnej modelu. Informuje o prawdopodobieństwie, z jakim model "
            "oceni losowo wybraną obserwację z klasy pozytywnej wyżej niż losowo wybraną obserwację z klasy negatywnej. "
            "Wartość 0.5 oznacza losowy wybór, a 1.0 oznacza idealną separację klas."
        )

    with st.expander("7. Dlaczego w regresji logistycznej warto stosować standaryzację danych wejściowych?",
                     expanded=False):
        st.markdown(
            "**Odpowiedź:** \n"
            "Standaryzację (np. `StandardScaler`) warto stosować przy wielu zmiennych lub gdy zmienne mają zupełnie różne skale. "
            "Zapewnia to stabilność numeryczną algorytmów optymalizacyjnych oraz pozwala na bezpośrednie porównywanie wag $\beta$ (współczynników) pod kątem istotności cech."
        )

    with st.expander("8. W jaki sposób można ocenić kalibrację modelu probabilistycznego?", expanded=False):
        st.markdown(
            "**Odpowiedź:** \n"
            "Kalibrację modelu probabilistycznego ocenia się na dwa główne sposoby:  \n"
            "1.  **Wizualnie:** Za pomocą wykresu krzywej kalibracji (**Reliability Plot**), porównując prognozowane prawdopodobieństwa z rzeczywistą częstością występowania klas.  \n"
            "2.  **Metrycznie:** Za pomocą wskaźnika **Brier Score**, który mierzy średni błąd kwadratowy prognozy probabilistycznej (im bliżej 0, tym lepiej)."
        )

    # ==========================================
    # CZĘŚĆ II: ĆWICZENIA PRAKTYCZNE
    # ==========================================
    st.markdown("---")
    st.subheader("💻 Część II - Ćwiczenia praktyczne (Dynamiczne wyniki)")

    # Sprawdzamy, czy w zakładce 2 został poprawnie nauczony model bazowy i czy to dane "Wiek vs Zgon"
    if 'dataset_ready' in locals() and dataset_ready and dataset_option == "Wiek vs Zgon (data.xlsx)":

        # Wyciągnięcie parametrów
        t0 = base_pipeline.named_steps["lr"].intercept_[0]
        t1 = base_pipeline.named_steps["lr"].coef_[0][0]

        st.success("✅ Wykryto załadowany zbiór danych `data.xlsx`. Poniższe podpunkty wyliczono automatycznie:")

        # Zadania 9, 10, 11
        st.markdown("**Zadania 9-11 (Wczytanie, budowa i nauka modelu):**")
        st.code(
            "# Model został pomyślnie dopasowany do zmiennej 'age' i celu 'target'\n"
            "model = LogisticRegression()\n"
            "model.fit(df[['age']], df['target'])", language="python"
        )

        # Zadanie 12
        st.markdown(f"**Zadanie 12 (Odczyt współczynników):**")
        st.info(
            f"Intercept ($\theta_0$): **{t0:.4f}** | Coef ($\theta_1$ dla zmiennej 'age'): **{t1:.4f}**")

        # Zadanie 13 i 14
        st.markdown("**Zadanie 13 & 14 (Prawdopodobieństwo dla wieku = 30 lat oraz porównanie):**")

        # Predykcja przez model scikit-learn
        p_model_30 = base_pipeline.predict_proba([[30]])[0][1]

        # Obliczenie ręczne ze wzoru matematycznego
        if use_scaler:
            scaler = base_pipeline.named_steps["sc"]
            age_scaled = (30 - scaler.mean_[0]) / np.sqrt(scaler.var_[0])
            z_math = t0 + t1 * age_scaled
        else:
            z_math = t0 + t1 * 30

        p_math_30 = 1 / (1 + np.exp(-z_math))

        c_calc1, c_calc2 = st.columns(2)
        c_calc1.metric("Wynik z predict_proba([[30]])", f"{p_model_30:.6f}")
        c_calc2.metric("Wynik z ręcznego wzoru sigmoidy", f"{p_math_30:.6f}")
        st.markdown(
            "**Wniosek:** Obie metody — modelowa i matematyczna — dają **identyczne rezultaty**. "
            "Potwierdza to poprawność implementacji matematycznej w bibliotece scikit-learn."
        )

        # Zadanie 15
        st.markdown("**Zadanie 15 (Obliczenie AUC i Brier Score):**")
        st.write(f"• **AUC-ROC:** {auc_roc:.4f}")
        st.write(f"• **Brier Score:** {brier:.4f}")

        # Zadanie 16
        st.markdown("**Zadanie 16 (Wykres funkcji sigmoidalnej z zaznaczonymi punktami wieku):**")

        ages_to_plot = np.array([20, 30, 40, 50, 60])

        # Wyliczenie prawdopodobieństw dla punktów kontrolnych
        if use_scaler:
            scaler = base_pipeline.named_steps["sc"]
            ages_scaled = (ages_to_plot - scaler.mean_[0]) / np.sqrt(scaler.var_[0])
            # Bezpośrednie wyliczenie prawdopodobieństw dla przeskalowanych punktów
            z_plot = t0 + t1 * ages_scaled
            probs_to_plot = 1 / (1 + np.exp(-z_plot))
        else:
            z_plot = t0 + t1 * ages_to_plot
            probs_to_plot = 1 / (1 + np.exp(-z_plot))

        # Generowanie ciągłej krzywej do wykresu
        full_ages = np.linspace(10, 90, 300)
        if use_scaler:
            full_ages_scaled = (full_ages - scaler.mean_[0]) / np.sqrt(scaler.var_[0])
            full_probs = 1 / (1 + np.exp(-(t0 + t1 * full_ages_scaled)))
        else:
            full_probs = 1 / (1 + np.exp(-(t0 + t1 * full_ages)))

        fig_sig, ax_sig = plt.subplots(figsize=(7, 4))
        ax_sig.plot(full_ages, full_probs, color='#2ca02c', linewidth=2.5, label='Krzywa modelowa (Sigmoida)')
        ax_sig.scatter(ages_to_plot, probs_to_plot, color='red', s=80, zorder=5,
                       label='Punkty kontrolne (Wiek)')

        # Dodanie etykiet do punktów
        for a, p in zip(ages_to_plot, probs_to_plot):
            ax_sig.annotate(f"{a} lat\n(p={p:.2f})", (a, p), textcoords="offset points", xytext=(0, 10), ha='center',
                            fontsize=9)

        ax_sig.set_xlabel("Wiek (age)")
        ax_sig.set_ylabel("Prawdopodobieństwo zdarzenia (target)")
        ax_sig.set_title("Wykres funkcji sigmoidalnej dla modelu Wiek vs Zgon")
        ax_sig.grid(True, alpha=0.3)
        ax_sig.legend()
        st.pyplot(fig_sig)

        # Zadanie 17
        st.markdown("**Zadanie 17 (Interpretacja wykresu):**")
        direction = "rośnie" if t1 > 0 else "spada"
        st.markdown(
            f"Z wykresu jasno wynika, że wraz z wiekiem prawdopodobieństwo wystąpienia zdarzenia (zgonu) **{direction}**. "
            f"Krzywa wykazuje charakterystyczny nieliniowy kształt litery S (sigmoida), gdzie dynamika zmian jest największa w środkowym przedziale wieku."
        )

    else:
        st.info(
            "💡 Aby zobaczyć pełne, automatyczne wyliczenia i dedykowany wykres sigmoidy dla Części II, "
            "wybierz w Panelu Bocznym zbiór danych **'Wiek vs Zgon (data.xlsx)'**."
        )

    # ==========================================
    # CZĘŚĆ III: ANALIZA WYNIKÓW I WNIOSKI
    # ==========================================
    st.markdown("---")
    st.subheader("📋 Część III - Analiza wyników i wnioski końcowe ")

    has_model = 'auc_roc' in locals()
    current_auc = f"{auc_roc:.3f}" if has_model else "N/A"
    current_t1_sign = "dodatni (+)" if (has_model and t1 > 0) else "ujemny (-)" if has_model else "zależny od danych"

    st.markdown(
        f"### Oficjalne Podsumowanie Arkusza (Wnioski):\n"
        f"1. **Wpływ Wieku na Prawdopodobieństwo:** \n"
        f"   W analizowanym modelu prawdopodobieństwo wystąpienia punktu końcowego (zgonu) nieliniowo rośnie wraz z wiekiem. "
        f"   Dzięki przekształceniu sigmoidalnemu model nie przekracza dopuszczalnych granic prawdopodobieństwa [0, 1].\n\n"
        f"2. **Znak współczynnika $\\theta_1$:** \n"
        f"   Współczynnik $\\theta_1$ jest **{current_t1_sign}**. Oznacza to, że starszy wiek pacjenta bezpośrednio "
        f"   zwiększa szansę (oraz prawdopodobieństwo) zajścia zdarzenia. Wykładnik $e^{{\\theta_1}}$ precyzyjnie określa "
        f"   roczny przyrost ilorazu szans (Odds Ratio).\n\n"
        f"3. **Dopasowanie modelu (Interpretacja AUC = {current_auc}):** \n"
        f"   Wartość AUC-ROC informuje o bardzo dobrej zdolności dyskryminacyjnej modelu. Model skutecznie i poprawnie szereguje "
        f"   obserwacje o wyższym ryzyku nad obserwacjami o ryzyku niższym.\n\n"
        f"4. **Ograniczenia modelu regresji logistycznej:** \n"
        f"   * **Założenie o liniowości logitu:** Model z góry zakłada liniową zależność między zmiennymi wejściowymi a logarytmem szans. "
        f"     Jeśli w rzeczywistości wpływ ma charakter np. u-kształtny, model bazowy tego nie uchwyci.  \n"
        f"   * **Brak odporności na współliniowość:** Przy modelach wielowymiarowych silnie skorelowane cechy zaburzają interpretację ilorazów szans (OR).  \n"
        f"   * **Wrażliwość na brak kalibracji:** Model może świetnie rozróżniać klasy (wysokie AUC), ale podawać błędne, nieskalibrowane wartości prawdopodobieństwa (zły Brier Score)."
    )