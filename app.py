import streamlit as st
import requests
import pandas as pd
import time
import os
import shutil
import tempfile
import re

# ==============================================================================
# 🛠️ CONFIGURATION
# ==============================================================================

DATASETS_MAP = {
    "Diplômes et Formation": "DS_RP_DIPLOMES_PRINC",
    "Éducation et Scolarisation": "DS_RP_EDUCATION",
    "Populations de référence": "DS_POPULATIONS_REFERENCE",
    "Historique Population (1968-2023)": "DS_POPULATIONS_HISTORIQUES",
    "Salaires Privé (Sexe & PCS)": "DS_BTS_SAL_EQTP_SEX_PCS",
    "Salaires Privé (Sexe & Âge)": "DS_BTS_SAL_EQTP_SEX_AGE",
    "Niveau de vie & Pauvreté (Âge)": "DS_FILOSOFI_AGE_TP_NIVVIE",
    "Niveau de vie & Pauvreté (Logement)": "DS_FILOSOFI_LOG_TP_NIVVIE",
    "Niveau de vie & Pauvreté (Type Ménage)": "DS_FILOSOFI_MEN_TP_NIVVIE",
    "Pauvreté : Indicateurs Principaux": "DS_FILOSOFI_CC",
    "Création d'Entreprises (Secteur)": "DS_SIDE_CREA_ENT_COM",
    "Création d'Établissements": "DS_SIDE_CREA_ETAB_COM",
    "Stocks Établissements (A10)": "DS_SIDE_STOCKS_ET_COM",
    "Stocks Unités Légales (A10)": "DS_SIDE_STOCKS_UL_COM",
    "Particuliers Employeurs": "DS_FLORES_PE",
    "Établissements (Sphères Économie)": "DS_FLORES_ECONOMIC_SPHERE",
    "Établissements Salariés (5 Secteurs)": "DS_FLORES_A5",
    "Établissements Salariés (17 Secteurs)": "DS_FLORES_A17",
    "Établissements Salariés (38 Secteurs)": "DS_FLORES_A38",
    "Établissements Salariés (88 Secteurs)": "DS_FLORES_A88",
    "Tourisme (Capacités Hébergement)": "DS_TOUR_CAP",
    "État Civil : Décès": "DS_ETAT_CIVIL_DECES_COMMUNES",
    "État Civil : Naissances": "DS_ETAT_CIVIL_NAIS_COMMUNES",
    "Équipements (Commerce, Services, Santé)": "DS_BPE",
    "Équipements (Sport, Loisirs, Culture)": "DS_BPE_SPORT_CULTURE",
    "Équipements (Enseignement)": "DS_BPE_EDUCATION",
    "Population (Principal)": "DS_RP_POPULATION_PRINC",
    "Population (Complémentaire)": "DS_RP_POPULATION_COMP",
    "Logements (Principal)": "DS_RP_LOGEMENT_PRINC",
    "Logements (Complémentaire)": "DS_RP_LOGEMENT_COMPL",
    "Ménages & Couples (Principal)": "DS_RP_MENAGES_PRINC",
    "Ménages (Complémentaire)": "DS_RP_MENAGES_COMP",
    "Familles (Complémentaire)": "DS_RP_FAMILLE_COMP",
    "Caractéristiques de l'Emploi (Princ)": "DS_RP_ACTIVITE_PRINC",
    "Chômage & Pop. Active (Princ)": "DS_RP_EMPLOI_LR_PRINC",
    "Chômage & Pop. Active (Comp)": "DS_RP_EMPLOI_LR_COMP",
    "Emploi au Lieu de Travail (Princ)": "DS_RP_EMPLOI_LT_PRINC",
    "Emploi au Lieu de Travail (Comp)": "DS_RP_EMPLOI_LT_COMP",
    "Navettes Domicile-Travail": "DS_RP_NAVETTES_PRINC",
    "Migrations Résidentielles": "DS_RP_MIGRES_PRINC",
    "Série Historique Recensement": "DS_RP_SERIE_HISTORIQUE",
    "Corps Électoral": "DS_ELECTORAL"
}

GEO_API_URL = "https://geo.api.gouv.fr"
INSEE_API_URL = "https://api.insee.fr/melodi/data"
MILLESIME_GEO = "2024"
REFERENCES_DIR = "references"

# Rate Limiting
TIME_BETWEEN_CALLS = 1.5 
PAUSE_ON_ERROR_429 = 60

# ==============================================================================
# 🧠 MOTEUR LOGIQUE
# ==============================================================================

def get_safe(url, params=None, headers=None):
    """Requête sécurisée avec gestion des quotas"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, headers=headers)
            if r.status_code == 200:
                time.sleep(TIME_BETWEEN_CALLS)
                return r
            elif r.status_code == 429:
                time.sleep(PAUSE_ON_ERROR_429)
                continue
            else:
                time.sleep(TIME_BETWEEN_CALLS)
                return r
        except:
            time.sleep(5)
    return None

class LocalReferenceEngine:
    """Gestion des traductions"""
    def __init__(self, ref_dir):
        self.ref_dir = ref_dir
        self.mappings = {}
        self.load_references()

    def load_references(self):
        if not os.path.exists(self.ref_dir): return
        files = [f for f in os.listdir(self.ref_dir) if f.endswith('.csv')]
        for filename in files:
            key = os.path.splitext(filename)[0]
            try:
                ref_df = pd.read_csv(os.path.join(self.ref_dir, filename), sep=';', dtype=str)
                if not ref_df.empty: ref_df.iloc[:, 0] = ref_df.iloc[:, 0].str.strip()
                if ref_df.shape[1] >= 2:
                    self.mappings[key] = pd.Series(ref_df.iloc[:, 1].values, index=ref_df.iloc[:, 0]).to_dict()
            except: pass

    def translate(self, df):
        df_out = df.copy()
        for col in df_out.columns:
            parts = col.split('.')
            mapping_key = next((p for p in parts if p in self.mappings), None)
            
            if mapping_key:
                mapping = self.mappings[mapping_key]
                df_out[col] = df_out[col].astype(str)
                if mapping_key == "GEO":
                    # On garde uniquement le code après le tiret
                    df_out[col] = df_out[col].apply(lambda x: x.split('-', 1)[1] if '-' in x else x).str.strip()
                df_out[col] = df_out[col].map(mapping).fillna(df_out[col])
        
        cols_to_rename = {c: "VALEUR" for c in df_out.columns if "OBS_VALUE" in c}
        if cols_to_rename: df_out.rename(columns=cols_to_rename, inplace=True)
        return df_out

def get_geo_targets(start_city_name, mode="SINGLE"):
    """
    Détermine la liste de toutes les zones à télécharger.
    """
    # 1. Trouver la ville de départ
    r = get_safe(f"{GEO_API_URL}/communes", params={"nom": start_city_name, "fields": "code,nom,codeEpci,codeRegion", "boost": "population"})
    if not r or not r.json():
        return None, "Ville introuvable"

    c = r.json()[0]
    targets = []
    epci_name = "EPCI_Inconnu"

    # 2. Logique selon le mode
    if mode == "EPCI":
        if not c.get('codeEpci'):
            return None, f"La commune de {c['nom']} n'appartient à aucun EPCI."
        
        # Récupérer le nom de l'EPCI pour le dossier
        r_epci_info = get_safe(f"{GEO_API_URL}/epcis/{c['codeEpci']}")
        if r_epci_info and r_epci_info.json():
            epci_name = r_epci_info.json().get('nom', c['codeEpci'])
        
        # Récupérer toutes les communes de l'EPCI
        r_communes = get_safe(f"{GEO_API_URL}/epcis/{c['codeEpci']}/communes")
        if r_communes:
            for com in r_communes.json():
                targets.append({
                    "param": f"{MILLESIME_GEO}-COM-{com['code']}",
                    "nom": com['nom'],
                    "type": "Commune"
                })
        
        # Ajouter l'EPCI lui-même (la moyenne globale)
        targets.append({"param": f"{MILLESIME_GEO}-EPCI-{c['codeEpci']}", "nom": epci_name, "type": "EPCI_Global"})
        
        folder_name = f"Export_{epci_name.replace(' ', '_')}"

    else: # Mode SINGLE (Ville seule)
        targets.append({"param": f"{MILLESIME_GEO}-COM-{c['code']}", "nom": c['nom'], "type": "Commune"})
        if c.get('codeEpci'):
            targets.append({"param": f"{MILLESIME_GEO}-EPCI-{c['codeEpci']}", "nom": "EPCI", "type": "EPCI"})
        
        folder_name = f"Export_{c['nom']}"

    # Ajout Région et France
    targets.append({"param": f"{MILLESIME_GEO}-REG-{c['codeRegion']}", "nom": "Region", "type": "Region"})
    targets.append({"param": f"{MILLESIME_GEO}-FRANCE-FM", "nom": "France Metro", "type": "France"})

    return targets, folder_name

def process_data_batched(targets, folder_name, selected_datasets_ids):
    """Récupère les données par paquets"""
    
    translator = LocalReferenceEngine(REFERENCES_DIR)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        final_dir = os.path.join(temp_dir, folder_name)
        os.makedirs(final_dir)
        status_log = []

        BATCH_SIZE = 20
        all_geo_params = [t['param'] for t in targets]
        
        batches = [all_geo_params[i:i + BATCH_SIZE] for i in range(0, len(all_geo_params), BATCH_SIZE)]
        
        progress_bar = st.progress(0)
        total_steps = len(selected_datasets_ids) * len(batches)
        current_step = 0

        for dataset_name, dataset_id in selected_datasets_ids.items():
            
            dataset_frames = []
            
            for batch in batches:
                params_list = [('GEO', code) for code in batch]
                
                url = f"{INSEE_API_URL}/{dataset_id}"
                r = get_safe(url, params=params_list, headers={'Accept': 'application/json'})
                
                if r and r.status_code == 200:
                    data = r.json().get("observations", [])
                    if data:
                        df_chunk = pd.json_normalize(data)
                        dataset_frames.append(df_chunk)
                
                current_step += 1
                progress_bar.progress(min(current_step / total_steps, 1.0))

            if dataset_frames:
                full_df = pd.concat(dataset_frames, ignore_index=True)
                full_df = translator.translate(full_df)
                
                code_to_name = {t['param'].split('-')[-1]: t['nom'] for t in targets}
                
                geo_col = next((c for c in full_df.columns if "GEO" in c), None)
                if geo_col:
                     full_df['LIBELLE_GEOGRAPHIQUE'] = full_df[geo_col].apply(
                         lambda x: code_to_name.get(str(x).split('-')[-1], x)
                     )
                     cols = ['LIBELLE_GEOGRAPHIQUE'] + [c for c in full_df.columns if c != 'LIBELLE_GEOGRAPHIQUE']
                     full_df = full_df[cols]

                filename = f"{dataset_name.replace(' ', '_')}.csv"
                full_df.to_csv(os.path.join(final_dir, filename), sep=";", index=False, encoding="utf-8-sig")
                status_log.append(f"✅ {dataset_name} : {len(full_df)} lignes")
            else:
                status_log.append(f"⚠️ {dataset_name} : Aucune donnée")

        archive_path = shutil.make_archive(os.path.join(temp_dir, folder_name), 'zip', root_dir=temp_dir, base_dir=folder_name)
        with open(archive_path, "rb") as f:
            zip_data = f.read()
            
    return zip_data, status_log

# ==============================================================================
# 🎨 INTERFACE
# ==============================================================================

st.set_page_config(page_title="Insee Extractor Pro", page_icon="🏙️")

st.markdown("""
<style>
    .stButton>button { width: 100%; background-color: #0068c9; color: white; border-radius: 8px;}
    div[data-testid="stExpander"] { border: 1px solid #e0e0e0; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

st.title("🏙️ Insee Extractor Pro")
st.markdown("Analysez un territoire entier en un clic.")

with st.container():
    col1, col2 = st.columns([2, 1])
    with col1:
        city_input = st.text_input("Ville de référence", placeholder="Ex: Dunkerque")
    with col2:
        mode = st.radio("Périmètre", ["Ville seule", "Tout l'EPCI (Agglo)"], horizontal=True)

    # --- NOUVEAU : Case à cocher pour TOUT sélectionner ---
    select_all = st.checkbox("✅ Tout sélectionner")
    
    all_options = list(DATASETS_MAP.keys())
    
    # Logique de sélection par défaut
    if select_all:
        default_selection = all_options
    else:
        # Sélection de base réduite si la case n'est pas cochée
        default_selection = ["Diplômes et Formation", "Caractéristiques de l'Emploi (Princ)"]

    datasets = st.multiselect(
        "Thématiques à exporter", 
        all_options, 
        default=default_selection
    )

if st.button("Lancer l'extraction"):
    if not city_input:
        st.error("Indiquez une ville.")
    elif not datasets:
        st.error("Sélectionnez au moins une donnée.")
    else:
        target_ids = {k: DATASETS_MAP[k] for k in datasets}
        internal_mode = "EPCI" if "EPCI" in mode else "SINGLE"
        
        with st.spinner("Analyse du territoire et récupération des codes..."):
            targets, folder_name = get_geo_targets(city_input, mode=internal_mode)
        
        if targets is None:
            st.error(folder_name)
        else:
            st.info(f"📍 Cible identifiée : **{len(targets)} zones géographiques** à traiter.")
            
            zip_file, logs = process_data_batched(targets, folder_name, target_ids)
            
            st.success("Extraction terminée !")
            with st.expander("Journal d'exécution"):
                for l in logs: st.write(l)
            
            st.download_button(
                label=f"📥 Télécharger {folder_name}.zip",
                data=zip_file,
                file_name=f"{folder_name}.zip",
                mime="application/zip"
            )