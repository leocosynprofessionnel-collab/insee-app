import streamlit as st
import requests
import pandas as pd
import time
import os
import shutil
import tempfile
import re

# ==============================================================================
# 🛠️ CONFIGURATION DES DONNÉES
# ==============================================================================

DATASETS_MAP = {
    "Caractéristiques de l'Emploi (Princ)": "DS_RP_ACTIVITE_PRINC",
    "Chômage & Pop. Active (Comp)": "DS_RP_EMPLOI_LR_COMP",
    "Chômage & Pop. Active (Princ)": "DS_RP_EMPLOI_LR_PRINC",
    "Corps Électoral": "DS_ELECTORAL",
    "Création d'Entreprises (Secteur)": "DS_SIDE_CREA_ENT_COM",
    "Création d'Entreprises à l'échelle supra-communal": "DS_SIDE_CREA_DEP_REG_NAT",
    "Création d'Établissements": "DS_SIDE_CREA_ETAB_COM",
    "Diplômes et Formation": "DS_RP_DIPLOMES_PRINC",
    "Éducation et Scolarisation": "DS_RP_EDUCATION",
    "Emploi au Lieu de Travail (Princ)": "DS_RP_EMPLOI_LT_PRINC",
    "Équipements (Commerce, Services, Santé)": "DS_BPE",
    "Équipements (Enseignement)": "DS_BPE_EDUCATION",
    "Équipements (Sport, Loisirs, Culture)": "DS_BPE_SPORT_CULTURE",
    "Équipements - évolution": "DS_BPE_EVOLUTION",
    "Établissements (Sphères Économie)": "DS_FLORES_ECONOMIC_SPHERE",
    "Établissements Salariés (17 Secteurs)": "DS_FLORES_A17",
    "Établissements Salariés (38 Secteurs)": "DS_FLORES_A38",
    "Établissements Salariés (5 Secteurs)": "DS_FLORES_A5",
    "Établissements Salariés (88 Secteurs)": "DS_FLORES_A88",
    "État Civil : Décès": "DS_ETAT_CIVIL_DECES_COMMUNES",
    "État Civil : Naissances": "DS_ETAT_CIVIL_NAIS_COMMUNES",
    "Historique Population (1968-2023)": "DS_POPULATIONS_HISTORIQUES",
    "Logements (Principal)": "DS_RP_LOGEMENT_PRINC",
    "Ménages & Couples (Principal)": "DS_RP_MENAGES_PRINC",
    "Migrations Résidentielles": "DS_RP_MIGRES_PRINC",
    "Navettes Domicile-Travail": "DS_RP_NAVETTES_PRINC",
    "Niveau de vie & Pauvreté (Âge)": "DS_FILOSOFI_AGE_TP_NIVVIE",
    "Niveau de vie & Pauvreté (Logement)": "DS_FILOSOFI_LOG_TP_NIVVIE",
    "Niveau de vie & Pauvreté (Type Ménage)": "DS_FILOSOFI_MEN_TP_NIVVIE",
    "Particuliers Employeurs": "DS_FLORES_PE",
    "Pauvreté : Indicateurs Principaux": "DS_FILOSOFI_CC",
    "Population (Principal)": "DS_RP_POPULATION_PRINC",
    "Populations de référence": "DS_POPULATIONS_REFERENCE",
    "Salaires Privé (Sexe & Âge)": "DS_BTS_SAL_EQTP_SEX_AGE",
    "Salaires Privé (Sexe & PCS)": "DS_BTS_SAL_EQTP_SEX_PCS",
    "Série Historique Recensement": "DS_RP_SERIE_HISTORIQUE",
    "Stocks Établissements (A10)": "DS_SIDE_STOCKS_ET_COM",
    "Stocks Unités Légales (A10)": "DS_SIDE_STOCKS_UL_COM",
    "Tourisme (Capacités Hébergement)": "DS_TOUR_CAP"
}

# Mapping des Thématiques (Regroupement)
THEMES_MAP = {
    "🧑‍🤝‍🧑 Population & dynamiques démographiques": [
        "Population (Principal)", "Populations de référence", "Historique Population (1968-2023)",
        "Série Historique Recensement", "Migrations Résidentielles", 
        "État Civil : Naissances", "État Civil : Décès"
    ],
    "🏠 Ménages, logements & conditions résidentielles": [
        "Ménages & Couples (Principal)", "Logements (Principal)",
        "Niveau de vie & Pauvreté (Logement)", "Niveau de vie & Pauvreté (Type Ménage)"
    ],
    "🎓 Éducation, formation & capital humain": [
        "Diplômes et Formation", "Éducation et Scolarisation", "Équipements (Enseignement)"
    ],
    "💼 Emploi, activité & marché du travail": [
        "Caractéristiques de l'Emploi (Princ)", "Emploi au Lieu de Travail (Princ)",
        "Chômage & Pop. Active (Princ)", "Navettes Domicile-Travail"
    ],
    "🏭 Tissu économique & appareil productif": [
        "Établissements Salariés (5 Secteurs)", "Établissements Salariés (17 Secteurs)", 
        "Établissements Salariés (38 Secteurs)", "Établissements Salariés (88 Secteurs)",
        "Établissements (Sphères Économie)", "Particuliers Employeurs",
        "Stocks Établissements (A10)", "Stocks Unités Légales (A10)"
    ],
    "🚀 Entrepreneuriat & dynamique de création": [
        "Création d'Entreprises (Secteur)", "Création d'Établissements", 
        "Création d'Entreprises à l'échelle supra-communal"
    ],
    "💶 Revenus, salaires & niveaux de vie": [
        "Salaires Privé (Sexe & Âge)", "Salaires Privé (Sexe & PCS)",
        "Niveau de vie & Pauvreté (Âge)", "Pauvreté : Indicateurs Principaux"
    ],
    "🗳️ Citoyenneté & vie démocratique": [
        "Corps Électoral"
    ],
    "🏥🛍️ Équipements, services & qualité de vie": [
        "Équipements (Commerce, Services, Santé)", "Équipements (Sport, Loisirs, Culture)",
        "Équipements - évolution"
    ],
    "🧳 Tourisme & économie présentielle": [
        "Tourisme (Capacités Hébergement)"
    ]
}

GEO_API_URL = "https://geo.api.gouv.fr"
INSEE_API_URL = "https://api.insee.fr/melodi/data"
MILLESIME_GEO = "2025"
REFERENCES_DIR = "references"

TIME_BETWEEN_CALLS = 1.5 
PAUSE_ON_ERROR_429 = 60

# ==============================================================================
# 🧠 MOTEUR TECHNIQUE
# ==============================================================================

def get_safe(url, params=None, headers=None):
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
    def __init__(self, root_dir, dataset_id):
        self.target_dir = os.path.join(root_dir, dataset_id)
        self.mappings = {}
        self.load_references()

    def load_references(self):
        if not os.path.exists(self.target_dir): return
        files = [f for f in os.listdir(self.target_dir) if f.endswith('.csv')]
        for filename in files:
            key = os.path.splitext(filename)[0]
            try:
                ref_df = pd.read_csv(os.path.join(self.target_dir, filename), sep=';', dtype=str)
                if not ref_df.empty: ref_df.iloc[:, 0] = ref_df.iloc[:, 0].str.strip()
                if ref_df.shape[1] >= 2:
                    self.mappings[key] = pd.Series(ref_df.iloc[:, 1].values, index=ref_df.iloc[:, 0]).to_dict()
            except: pass

    def translate(self, df):
        df_out = df.copy()
        renames = {}
        for col in df_out.columns:
            parts = col.split('.')
            clean_name = parts[-1]
            if clean_name in self.mappings:
                mapping = self.mappings[clean_name]
                df_out[col] = df_out[col].astype(str)
                code_series = df_out[col]
                if clean_name == "GEO":
                    code_series = df_out[col].apply(lambda x: x.split('-', 1)[1] if '-' in x else x).str.strip()
                    df_out[col] = code_series
                
                libelle_col = f"LIBELLE_{clean_name}"
                df_out[libelle_col] = code_series.map(mapping).fillna(code_series)
                renames[col] = f"CODE_{clean_name}"
            elif "OBS_VALUE" in col: renames[col] = "VALEUR"
            elif "OBS_STATUS" in col: renames[col] = "STATUT_DONNEE"
            elif "dim" in parts[0] or "dimensions" in parts[0]: renames[col] = f"CODE_{clean_name}"

        if renames: df_out.rename(columns=renames, inplace=True)
        return df_out

# ==============================================================================
# 🌍 MOTEUR GÉOGRAPHIQUE AVANCÉ
# ==============================================================================

def search_territory(query):
    results = []
    r = get_safe(f"{GEO_API_URL}/communes", params={"nom": query, "fields": "nom,code,codeEpci,codeDepartement,codeRegion", "boost": "population", "limit": 5})
    if r and r.ok:
        for c in r.json():
            results.append({"label": f"🏙️ Commune : {c['nom']} ({c['code']})", "type": "Commune", "data": c, "id": f"COM-{c['code']}"})
    r = get_safe(f"{GEO_API_URL}/epcis", params={"nom": query, "fields": "nom,code", "limit": 3})
    if r and r.ok:
        for c in r.json():
            results.append({"label": f"⚙️ EPCI : {c['nom']}", "type": "EPCI", "data": c, "id": f"EPCI-{c['code']}"})
    r = get_safe(f"{GEO_API_URL}/departements", params={"nom": query, "fields": "nom,code,codeRegion", "limit": 3})
    if r and r.ok:
        for c in r.json():
            results.append({"label": f"🏛️ Département : {c['nom']} ({c['code']})", "type": "Département", "data": c, "id": f"DEP-{c['code']}"})
    r = get_safe(f"{GEO_API_URL}/regions", params={"nom": query, "fields": "nom,code", "limit": 3})
    if r and r.ok:
        for c in r.json():
            results.append({"label": f"🌍 Région : {c['nom']}", "type": "Région", "data": c, "id": f"REG-{c['code']}"})
    return results

def get_comparison_targets_dynamic(selected_item, comparisons):
    targets = []
    main_data = selected_item['data']
    t_type = selected_item['type']
    
    # Cible
    if t_type == "Commune": targets.append({"param": f"{MILLESIME_GEO}-COM-{main_data['code']}", "nom": main_data['nom'], "type": "Cible"})
    elif t_type == "EPCI": targets.append({"param": f"{MILLESIME_GEO}-EPCI-{main_data['code']}", "nom": main_data['nom'], "type": "Cible"})
    elif t_type == "Département": targets.append({"param": f"{MILLESIME_GEO}-DEP-{main_data['code']}", "nom": main_data['nom'], "type": "Cible"})
    elif t_type == "Région": targets.append({"param": f"{MILLESIME_GEO}-REG-{main_data['code']}", "nom": main_data['nom'], "type": "Cible"})

    # Pairs
    if "Communes du même EPCI" in comparisons and t_type == "Commune" and main_data.get('codeEpci'):
        r = get_safe(f"{GEO_API_URL}/epcis/{main_data['codeEpci']}/communes")
        if r and r.ok:
            for c in r.json():
                if c['code'] != main_data['code']: targets.append({"param": f"{MILLESIME_GEO}-COM-{c['code']}", "nom": c['nom'], "type": "Voisin EPCI"})
    if "Communes membres" in comparisons and t_type == "EPCI":
        r = get_safe(f"{GEO_API_URL}/epcis/{main_data['code']}/communes")
        if r and r.ok:
            for c in r.json(): targets.append({"param": f"{MILLESIME_GEO}-COM-{c['code']}", "nom": c['nom'], "type": "Membre EPCI"})
    if "Autres EPCI du Département" in comparisons:
        dept_code = main_data.get('codeDepartement') if t_type == "Commune" else None
        if t_type == "EPCI":
             r_d = get_safe(f"{GEO_API_URL}/epcis/{main_data['code']}", params={"fields": "codeDepartement"})
             if r_d and r_d.ok: dept_code = r_d.json().get('codeDepartement')
        if dept_code:
            r_e = get_safe(f"{GEO_API_URL}/departements/{dept_code}/epcis")
            if r_e and r_e.ok:
                for e in r_e.json():
                    if e['code'] != main_data.get('code') and e['code'] != main_data.get('codeEpci'):
                        targets.append({"param": f"{MILLESIME_GEO}-EPCI-{e['code']}", "nom": e['nom'], "type": "Voisin Département"})
    if "Autres Départements de la Région" in comparisons:
        reg_code = main_data.get('codeRegion')
        if reg_code:
            r_d = get_safe(f"{GEO_API_URL}/regions/{reg_code}/departements")
            if r_d and r_d.ok:
                for d in r_d.json():
                    if d['code'] != main_data.get('code') and d['code'] != main_data.get('codeDepartement'):
                        targets.append({"param": f"{MILLESIME_GEO}-DEP-{d['code']}", "nom": d['nom'], "type": "Voisin Région"})
    if "Toutes les Régions" in comparisons:
        r_r = get_safe(f"{GEO_API_URL}/regions")
        if r_r and r_r.ok:
            for reg in r_r.json():
                if reg['code'] != main_data.get('code') and reg['code'] != main_data.get('codeRegion'):
                    targets.append({"param": f"{MILLESIME_GEO}-REG-{reg['code']}", "nom": reg['nom'], "type": "Autre Région"})

    # Hiérarchie
    if "EPCI" in comparisons and (main_data.get('codeEpci') or t_type == "Commune"):
        c = main_data.get('codeEpci')
        if c: targets.append({"param": f"{MILLESIME_GEO}-EPCI-{c}", "nom": "EPCI", "type": "Parent"})
    if "Département" in comparisons:
        c = main_data.get('codeDepartement')
        if not c and t_type=="EPCI":
             r_d = get_safe(f"{GEO_API_URL}/epcis/{main_data['code']}", params={"fields": "codeDepartement"})
             if r_d and r_d.ok: c = r_d.json().get('codeDepartement')
        if c: targets.append({"param": f"{MILLESIME_GEO}-DEP-{c}", "nom": "Département", "type": "Parent"})
    if "Région" in comparisons and main_data.get('codeRegion'):
        targets.append({"param": f"{MILLESIME_GEO}-REG-{main_data['codeRegion']}", "nom": "Région", "type": "Parent"})
    if "France" in comparisons:
        targets.append({"param": f"{MILLESIME_GEO}-FRANCE-FM", "nom": "France Métropolitaine", "type": "National"})

    return targets, f"Export_{main_data['nom'].replace(' ', '_')}"

def process_data_batched(targets, folder_name, selected_datasets_ids):
    with tempfile.TemporaryDirectory() as temp_dir:
        final_dir = os.path.join(temp_dir, folder_name)
        os.makedirs(final_dir)
        status_log = []
        BATCH_SIZE = 5 
        all_geo_params = [t['param'] for t in targets]
        batches = [all_geo_params[i:i + BATCH_SIZE] for i in range(0, len(all_geo_params), BATCH_SIZE)]
        progress_bar = st.progress(0)
        total_steps = len(selected_datasets_ids) * len(batches)
        current_step = 0

        for dataset_name, dataset_id in selected_datasets_ids.items():
            translator = LocalReferenceEngine(REFERENCES_DIR, dataset_id)
            dataset_frames = []
            for batch in batches:
                params_list = [('GEO', code) for code in batch]
                url = f"{INSEE_API_URL}/{dataset_id}"
                r = get_safe(url, params=params_list, headers={'Accept': 'application/json'})
                if r and r.status_code == 200:
                    data = r.json().get("observations", [])
                    if data: dataset_frames.append(pd.json_normalize(data))
                current_step += 1
                progress_bar.progress(min(current_step / total_steps, 1.0))

            if dataset_frames:
                full_df = pd.concat(dataset_frames, ignore_index=True)
                full_df = translator.translate(full_df)
                code_to_name = {t['param'].split('-')[-1]: t['nom'] for t in targets}
                geo_col = next((c for c in full_df.columns if "CODE_GEO" in c), None)
                if geo_col:
                     full_df['LIBELLE_GEOGRAPHIQUE'] = full_df[geo_col].apply(lambda x: code_to_name.get(str(x), x))
                     first_cols = ['LIBELLE_GEOGRAPHIQUE', 'VALEUR']
                     libelle_cols = [c for c in full_df.columns if c.startswith('LIBELLE_') and c != 'LIBELLE_GEOGRAPHIQUE']
                     code_cols = [c for c in full_df.columns if c.startswith('CODE_')]
                     other_cols = [c for c in full_df.columns if c not in first_cols + libelle_cols + code_cols]
                     final_order = [c for c in first_cols if c in full_df.columns] + sorted(libelle_cols) + sorted(code_cols) + sorted(other_cols)
                     full_df = full_df[final_order]
                filename = f"{dataset_name.replace(' ', '_')}.csv"
                full_df.to_csv(os.path.join(final_dir, filename), sep=";", index=False, encoding="utf-8-sig")
                status_log.append(f"✅ {dataset_name} : {len(full_df)} lignes")
            else:
                status_log.append(f"⚠️ {dataset_name} : Aucune donnée")

        archive_path = shutil.make_archive(os.path.join(temp_dir, folder_name), 'zip', root_dir=temp_dir, base_dir=folder_name)
        with open(archive_path, "rb") as f: zip_data = f.read()
    return zip_data, status_log

# ==============================================================================
# 🎨 INTERFACE STREAMLIT
# ==============================================================================

st.set_page_config(page_title="Insee Extraction", page_icon="🌍")
st.markdown("""<style>.stButton>button { width: 100%; background-color: #0068c9; color: white; border-radius: 8px;} div[data-testid="stExpander"] { border: 1px solid #e0e0e0; border-radius: 8px; }</style>""", unsafe_allow_html=True)

st.title("🌍 Extracteur Universel des données de l'INSEE")
st.markdown("Recherchez un territoire, choisissez vos thématiques et comparez.")

# 1. RECHERCHE
with st.container():
    search_query = st.text_input("🔍 Rechercher un territoire", placeholder="Ex: Dunkerque, Nord, Gironde...")
    if 'search_results' not in st.session_state: st.session_state.search_results = []
    if st.button("Lancer la recherche"):
        with st.spinner("Recherche..."): st.session_state.search_results = search_territory(search_query)

# 2. SELECTION
selected_territory = None
if st.session_state.search_results:
    options = {item['label']: item for item in st.session_state.search_results}
    choice = st.selectbox("📍 Résultat exact :", list(options.keys()))
    selected_territory = options[choice]
    st.success(f"Territoire : **{selected_territory['type']} - {selected_territory['data']['nom']}**")

# 3. CONFIGURATION
if selected_territory:
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Comparaisons")
        t_type = selected_territory['type']
        comparisons = []
        
        st.caption("↔️ Pairs (Horizontal)")
        if t_type == "Commune":
            if st.checkbox("Toutes les communes de l'EPCI", value=True): comparisons.append("Communes du même EPCI")
        elif t_type == "EPCI":
            if st.checkbox("Autres EPCI du Département"): comparisons.append("Autres EPCI du Département")
            if st.checkbox("Détail de mes Communes membres", value=True): comparisons.append("Communes membres")
        elif t_type == "Département":
            if st.checkbox("Autres Départements de la Région"): comparisons.append("Autres Départements de la Région")
        elif t_type == "Région":
            if st.checkbox("Toutes les Régions de France"): comparisons.append("Toutes les Régions")
        
        st.caption("↕️ Hiérarchie (Vertical)")
        if t_type == "Commune":
            if st.checkbox("EPCI", value=True): comparisons.append("EPCI")
            if st.checkbox("Département"): comparisons.append("Département")
            if st.checkbox("Région"): comparisons.append("Région")
        elif t_type == "EPCI":
            if st.checkbox("Département"): comparisons.append("Département")
            if st.checkbox("Région"): comparisons.append("Région")
        elif t_type == "Département":
            if st.checkbox("Région"): comparisons.append("Région")
        if st.checkbox("France Métropolitaine", value=True): comparisons.append("France")

    with col2:
        st.subheader("2. Données")
        
        # --- NOUVEAU SELECTEUR DE THÈMES ---
        st.write("📂 **Sélection Rapide par Thématique**")
        selected_themes = st.multiselect("Ajouter des thématiques entières :", list(THEMES_MAP.keys()))
        
        # Case Tout Sélectionner
        select_all = st.checkbox("✅ Tout sélectionner (toutes les bases)")
        
        # Calcul de la sélection par défaut
        default_datasets = []
        if select_all:
            default_datasets = list(DATASETS_MAP.keys())
        elif selected_themes:
            # On ajoute les datasets des thèmes choisis
            for theme in selected_themes:
                default_datasets.extend(THEMES_MAP[theme])
            # On dédoublonne
            default_datasets = list(set(default_datasets))
        else:
            # Défaut minimal si rien n'est coché
            default_datasets = ["Diplômes et Formation", "Caractéristiques de l'Emploi (Princ)"]

        # Le Multiselect Final (modifiable par l'utilisateur)
        st.write("📝 **Ajuster la sélection précise :**")
        datasets = st.multiselect("Bases de données", list(DATASETS_MAP.keys()), default=default_datasets)

    st.divider()
    if st.button("🚀 Extraire les données"):
        if not datasets:
            st.error("Aucune donnée sélectionnée.")
        else:
            target_ids = {k: DATASETS_MAP[k] for k in datasets}
            with st.spinner("Traitement en cours..."):
                targets, folder_name = get_comparison_targets_dynamic(selected_territory, comparisons)
                st.info(f"📦 {len(targets)} zones géographiques.")
                zip_file, logs = process_data_batched(targets, folder_name, target_ids)
                st.success("Terminé !")
                st.download_button(f"📥 Télécharger {folder_name}.zip", zip_file, f"{folder_name}.zip", "application/zip")
                with st.expander("Détails"):
                     for l in logs: st.write(l)