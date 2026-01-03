import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. Titre et présentation de l'app
st.set_page_config(page_title="Antibio-Tracker", page_icon="🦠")
st.title("🦠 Antibio-Tracker : Prédictions de Résistance")
st.write("""
Cette application permet de visualiser l'évolution de la résistance aux antibiotiques 
en Europe et de prédire les tendances futures grâce à l'intelligence artificielle.
**Source des données :** ECDC (European Centre for Disease Prevention and Control).
""")

# 2. Chargement des données (On met ça en cache pour que ce soit rapide)
@st.cache_data
def load_data():
    # Remplacez par les vrais noms de vos fichiers s'ils sont dans le même dossier
    # Pour l'exemple, on imagine que vous avez fusionné vos 3 fichiers en un seul CSV final
    # Ou alors on charge les 3 ici comme dans votre Colab
    try:
        # Option A : Si vous avez un gros fichier fusionné
        # df = pd.read_csv("donnees_completes.csv")
        
        # Option B : On charge les 3 fichiers séparés (si vous les avez en local)
        # Note : Il faudra que ces fichiers soient dans le même dossier que ce script
        df1 = pd.read_csv("ecoli_data.csv") # Remplacez par le vrai nom
        df2 = pd.read_csv("staph_aureus_data.csv")
        df3 = pd.read_csv("klebsiella_pneumoniae_data.csv")
        df = pd.concat([df1, df2, df3], ignore_index=True)
        
        # Nettoyage
        df['NumValue'] = pd.to_numeric(df['NumValue'], errors='coerce')
        df = df.dropna(subset=['NumValue'])
        return df
    except FileNotFoundError:
        st.error("Erreur : Les fichiers CSV sont introuvables. Vérifiez qu'ils sont bien dans le dossier.")
        return pd.DataFrame()

df = load_data()

if not df.empty:
    # 3. La Barre Latérale (Les Menus)
    st.sidebar.header("Paramètres")
    
    # Menu Pays
    liste_pays = sorted(df['RegionName'].unique())
    pays = st.sidebar.selectbox("Choisissez un Pays", liste_pays)
    
    # Menu Bactérie
    liste_bacteries = sorted(df['Population'].unique())
    bacterie = st.sidebar.selectbox("Choisissez une Bactérie/Antibiotique", liste_bacteries)
    
    # 4. Le Cœur du Réacteur (Filtrage et Calculs)
    
    # Filtrage
    data = df[(df['RegionName'] == pays) & (df['Population'] == bacterie)]
    data = data[data['NumValue'] > 0] # On enlève les zéros
    
    if len(data) < 2:
        st.warning(f"⚠️ Pas assez de données fiables pour analyser {pays} / {bacterie}.")
    else:
        # IA (Régression)
        data = data.sort_values('Time')
        X = data['Time'].values.reshape(-1, 1)
        y = data['NumValue'].values
        
        modele = LinearRegression()
        modele.fit(X, y)
        
        annees_futures = np.array([[2024], [2025], [2026], [2028], [2030]])
        predictions = modele.predict(annees_futures)
        
        # 5. Affichage des Résultats
        
        # Colonnes pour afficher les chiffres clés joliment
        col1, col2 = st.columns(2)
        res_actuel = y[-1]
        res_2030 = max(0, predictions[-1]) # Pas de négatif
        
        with col1:
            st.metric("Dernier taux connu", f"{res_actuel:.1f}%")
        with col2:
            variation = res_2030 - res_actuel
            st.metric("Prédiction 2030", f"{res_2030:.1f}%", delta=f"{variation:.1f}%", delta_color="inverse")

        # Graphique
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(X, y, color='blue', label='Historique')
        ax.plot(X, modele.predict(X), color='green', alpha=0.5, linestyle='--', label='Tendance')
        ax.scatter(annees_futures, predictions, color='red', s=50, label='Prédiction IA')
        
        ax.set_title(f"Dynamique de résistance : {pays}")
        ax.set_xlabel("Année")
        ax.set_ylabel("Résistance (%)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # La commande magique pour afficher le graph dans l'app
        st.pyplot(fig)

        # Petit texte d'explication
        if variation > 0:
            st.warning("⚠️ La tendance est à la hausse. Une surveillance accrue est nécessaire.")
        else:
            st.success("✅ La tendance est à la baisse ou stable. Les mesures semblent efficaces.") 
            
st.markdown("---") # Ligne de séparation
st.caption("Réalisé par Raphaël Noyer | Projet Étudiant Biologie")
