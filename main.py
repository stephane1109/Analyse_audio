############
# Analyse de l'amplitude sonore dans un discours
# Stéphane Meurisse
# www.codeandcortex.fr
# Date : 19-04-2024
############

import streamlit as st
import numpy as np
import soundfile as sf
import plotly.graph_objects as go
import pandas as pd
import tempfile
import os


def convertir_en_min_sec(seconds):
    """
    Convertit un temps (en secondes) en format mm:ss.
    """
    minutes = int(seconds // 60)
    sec = int(seconds % 60)
    return f"{minutes:02d}:{sec:02d}"


def transcrire_audio_whisper(uploaded_file):
    """
    Transcrit le fichier audio uploadé avec Whisper.
    Le fichier audio est sauvegardé temporairement pour la transcription.

    Returns:
        list: Liste des segments de transcription avec 'start', 'end' et 'text'.
    """
    try:
        import whisper
    except ImportError:
        st.error("Le module 'whisper' n'est pas installé. Installez-le avec 'pip install -U openai-whisper'.")
        return []

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_audio_path = temp_file.name

    model = whisper.load_model("small")
    result = model.transcribe(temp_audio_path, language="fr")
    os.remove(temp_audio_path)
    return result.get("segments", [])


def downsample_by_second(data, times, samplerate):
    """
    Regroupe le signal par intervalles fixes de 1 seconde.

    Chaque bin correspond exactement aux échantillons d'une seconde (soit samplerate échantillons).
    Pour chaque intervalle (bin), on calcule :
      - Le temps moyen (généralement approximativement la moitié de la seconde)
      - La valeur minimale et la valeur maximale dans cet intervalle (bin).

    Args:
        data (np.array): Tableau du signal audio.
        times (np.array): Vecteur temps du signal.
        samplerate (int): Nombre d'échantillons par seconde.

    Returns:
        times_bins (np.array): Temps moyen par intervalle (bin)
        min_vals (np.array): Valeurs minimales par intervalle (bin)
        max_vals (np.array): Valeurs maximales par intervalle (bin)
        avg_vals (np.array): Valeur moyenne par intervalle, calculée ici comme la moyenne de min et max
    """
    n = len(data)
    bin_size = samplerate  # chaque bin est 1 seconde
    nb_bins = n // bin_size  # on ignore l'excédent si la durée n'est pas entière

    times_bins = []
    min_vals = []
    max_vals = []
    avg_vals = []

    for i in range(nb_bins):
        start = i * bin_size
        end = start + bin_size
        times_bins.append(np.mean(times[start:end]))
        min_vals.append(np.min(data[start:end]))
        max_vals.append(np.max(data[start:end]))
        avg_vals.append((np.min(data[start:end]) + np.max(data[start:end])) / 2.0)

    return np.array(times_bins), np.array(min_vals), np.array(max_vals), np.array(avg_vals)


# --- Interface Streamlit ---
st.title("Analyse de l'amplitude sonore dans un discours")

st.markdown("""
### Introduction
Ce script analyse un fichier audio (.wav) en regroupant le signal en intervalles fixes de 1 seconde.
Chaque intervalle (bin) (correspondant à 1 seconde) contient un ensemble d'échantillons. Pour chaque intervalle, 
on calcule :
- Le **temps moyen** de l'intervalle,
- La **valeur minimale** (creux) et la **valeur maximale** (pic) du signal dans cet intervalle,
- La **valeur moyenne** de l'intervalle (moyenne de min et max).

Ces statistiques permettent d'obtenir une représentation condensée du signal sur une base temporelle régulière.
L'analyse statistique (calcul de la moyenne globale et de l'écart‑type) se fait ensuite sur le signal résumé (les valeurs moyennes par intervalle (bin)).
Nous définissons un intervalle de détection [μ ± k×σ] pour repérer les observations atypiques.
Enfin, si la transcription est activée, le script associe à chaque intervalle (bin) atypique le segment de texte correspondant sur la fenêtre [t−1, t+1].
""")

# Téléchargement du fichier audio
uploaded_file = st.file_uploader("Importer un fichier audio (.wav)", type=["wav"])

# Option transcription
afficher_transcription = st.checkbox("Afficher la transcription avec Whisper (pour le concordancier)", value=False)

# Paramètre k pour l'intervalle [μ ± k×σ]
k_value = st.slider("Définissez le paramètre k (pour l'intervalle [μ ± k×σ])", min_value=1.0, max_value=5.0, value=2.0,
                    step=0.1)

if st.button("Lancer l'analyse"):
    if uploaded_file is None:
        st.info("Veuillez importer un fichier audio (WAV).")
        st.stop()

    try:
        data, samplerate = sf.read(uploaded_file)
        st.write(f"Taux d'échantillonnage : {samplerate} Hz")
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier audio : {e}")
        st.stop()

    # Conversion en mono si nécessaire
    if data.ndim > 1:
        data = data.mean(axis=1)

    n_samples = len(data)
    duration = n_samples / samplerate
    st.info(f"Durée du fichier audio : **{duration:.2f} secondes** ({n_samples} observations).")

    # Création du vecteur temps complet
    temps_complet = np.linspace(0, duration, n_samples)

    # Regroupement par bins d'1 seconde
    times_bins, min_vals, max_vals, avg_vals = downsample_by_second(data, temps_complet, samplerate)

    # Affichage du nombre de bins
    st.info(f"Le signal est divisé en {len(times_bins)} intervalles (intervalle = 1 seconde).")

    # Calcul de la moyenne globale et de l'écart‑type sur les valeurs moyennes par bin
    mu = np.mean(avg_vals)
    sigma = np.std(avg_vals)
    st.write(f"Moyenne (μ) : {mu:.4f}")
    st.write(f"Écart‑type (σ) : {sigma:.4f}")

    lower_bound = mu - k_value * sigma
    upper_bound = mu + k_value * sigma
    st.markdown(f"Intervalle de détection [μ−k×σ,μ+k×σ] : **[{lower_bound:.4f}, {upper_bound:.4f}]**")

    # Détection des bins atypiques sur la base de la valeur moyenne
    # On détermine les indices des bins dont la valeur moyenne (avg_vals) est en dehors de [μ ± k×σ]
    indices_outliers = np.where((avg_vals < lower_bound) | (avg_vals > upper_bound))[0]
    times_out = times_bins[indices_outliers]
    avg_out = avg_vals[indices_outliers]

    # Affichage du nombre de bins atypiques
    st.info(f"Nombre d'observations atypiques détectés pour k={k_value} : {len(indices_outliers)}")

    #### Graphique 1 : Affichage du signal par bins
    # Nous allons afficher une "enveloppe" qui représente les valeurs minimales et maximales pour chaque bin, et superposer les valeurs moyennes.
    fig_bins = go.Figure()
    # Zone d'enveloppe représentant le min et le max dans chaque bin (en jaune clair)
    fig_bins.add_trace(go.Scatter(
        x=np.concatenate([times_bins, times_bins[::-1]]),
        y=np.concatenate([min_vals, max_vals[::-1]]),
        fill="toself",
        fillcolor="rgba(255,255,0,0.2)",  # jaune clair
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        name="Enveloppe (min/max)"
    ))
    # Valeur moyenne par bin (en bleu)
    fig_bins.add_trace(go.Scatter(
        x=times_bins,
        y=avg_vals,
        mode="lines+markers",
        name="Signal (moyenne par intervalle)",
        marker=dict(color="blue"),
        line=dict(color="blue")
    ))
    # Marquage des bins atypiques (avec valeur moyenne en rouge)
    if len(indices_outliers) > 0:
        fig_bins.add_trace(go.Scatter(
            x=times_out,
            y=avg_out,
            mode="markers",
            marker=dict(color="red", size=8, symbol="diamond"),
            name="Observations atypiques"
        ))
    fig_bins.update_layout(
        title="Signal regroupé par intervalle d'1 seconde",
        xaxis_title="Temps (s)",
        yaxis_title="Amplitude",
        width=800,
        height=500,
        hovermode="x unified"
    )
    st.plotly_chart(fig_bins, use_container_width=True)

    #### Option de transcription pour ajouter des segments de texte au concordancier
    if afficher_transcription:
        st.info("Transcription en cours (cela peut prendre quelques minutes)...")
        transcription_segments = transcrire_audio_whisper(uploaded_file)
        if len(transcription_segments) == 0:
            st.warning("Aucun segment de transcription n'a été généré.")
    else:
        transcription_segments = []

    #### Concordancier des bins atypiques avec segments de texte
    st.subheader("Concordancier des observations atypiques")
    if len(indices_outliers) > 0:
        concordance = []
        for t, avg in zip(times_out, avg_out):
            # Extraction du texte dans la fenêtre [t-1, t+1]
            segment_text = ""
            if transcription_segments:
                segment_text = " ".join(
                    seg["text"].strip()
                    for seg in transcription_segments
                    if seg["end"] >= t - 1 and seg["start"] <= t + 1
                )
            concordance.append({
                "Timestamp (s)": f"{t:.3f}",
                "Time (mm:ss)": convertir_en_min_sec(t),
                "Valeur moyenne": f"{avg:.4f}",
                "Segment texte": segment_text
            })
        df_concordance = pd.DataFrame(concordance)
        st.dataframe(df_concordance)
        st.download_button(
            label="Télécharger le concordancier en CSV",
            data=df_concordance.to_csv(index=False).encode("utf-8"),
            file_name="concordancier.csv",
            mime="text/csv"
        )
    else:
        st.info("Aucune observation atypique n'a été détecté.")

    st.markdown("""
    **Interprétation générale :**
    - **Graphique :**  
      Le signal audio est regroupé par intervalle d'1 seconde. Chaque intervalle affiche une enveloppe (la valeur minimale et maximale) et une valeur moyenne (en bleu).  
      Les intervalles dont la valeur moyenne sort de l'intervalle de confiance [μ ± k×σ] (défini ici pour le signal résumé par les moyennes des intervalles) sont marqués en rouge.
    - **Concordancier :**  
      Le tableau récapitule, pour chaque observation atypique, son timestamp (en secondes et au format mm:ss), la valeur moyenne, et, si la transcription est activée, le segment de texte correspondant à la fenêtre [t-1, t+1].
    """)
else:
    st.info("Veuillez importer un fichier audio (.wav).")

