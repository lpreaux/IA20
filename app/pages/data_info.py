import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, DataReturnMode, GridUpdateMode

from dataset.state import DatasetState
from dataset.forms.config import dataset_config_form


def display_dataset_overview(dataset):
    """Affiche une vue d'ensemble du dataset avec des métriques clés."""
    st.markdown("""
    # 📊 Vue d'ensemble du jeu de données

    Explorez et configurez votre jeu de données. Cette page vous permet de :
    - Visualiser l'aperçu complet des données
    - Gérer la sélection des variables explicatives et cibles
    - Consulter les statistiques détaillées
    """)

    # Métriques principales dans des colonnes
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Observations",
            f"{len(dataset.data):,}",
            help="Nombre total de lignes dans le dataset"
        )
    with col2:
        st.metric(
            "Variables",
            len(dataset.data.columns),
            f"{len(dataset.features_columns)} features",
            help="Nombre total de colonnes"
        )
    with col3:
        st.metric(
            "Variables cibles",
            len(dataset.target_columns),
            help="Nombre de variables à prédire"
        )
    with col4:
        missing_data = dataset.data.isnull().sum().sum()
        missing_percentage = (missing_data / (dataset.data.size)) * 100
        st.metric(
            "Données manquantes",
            f"{missing_percentage:.1f}%",
            help="Pourcentage de valeurs manquantes dans le dataset"
        )


def display_data_quality_insights(dataset):
    """Affiche des insights sur la qualité des données."""
    st.markdown("## 📈 Qualité des données")

    # Analyses par colonne
    quality_data = []
    for col in dataset.data.columns:
        missing = dataset.data[col].isnull().sum()
        missing_pct = (missing / len(dataset.data)) * 100
        unique_values = dataset.data[col].nunique()
        unique_pct = (unique_values / len(dataset.data)) * 100

        col_type = "🎯 Target" if col in dataset.target_columns else "✨ Feature" if col in dataset.features_columns else "📎 Non utilisée"

        quality_data.append({
            "Colonne": col,
            "Type": col_type,
            "Type de données": str(dataset.data[col].dtype),
            "Valeurs uniques": f"{unique_values:,} ({unique_pct:.1f}%)",
            "Valeurs manquantes": f"{missing:,} ({missing_pct:.1f}%)",
        })

    quality_df = pd.DataFrame(quality_data)

    # Configuration du style pour l'affichage
    def highlight_type(val):
        if "Target" in val:
            return 'background-color: rgba(255, 99, 71, 0.2)'
        elif "Feature" in val:
            return 'background-color: rgba(46, 139, 87, 0.2)'
        return ''

    st.dataframe(
        quality_df.style.applymap(highlight_type, subset=['Type']),
        use_container_width=True,
        height=400
    )


def configure_grid_display(dataset):
    """Configure et affiche la grille de données interactive."""
    st.markdown("## 🔍 Aperçu détaillé des données")

    gb = GridOptionsBuilder.from_dataframe(dataset.data)

    # Style personnalisé pour la grille
    grid_style = {
        "cssText": """
            .ag-theme-streamlit {
                --ag-header-height: 60px;
                --ag-header-foreground-color: #FFFFFF;
                --ag-header-background-color: #1E1E1E;
                --ag-row-hover-color: rgba(255, 255, 255, 0.1);
                --ag-selected-row-background-color: rgba(255, 255, 255, 0.2);
                --ag-font-size: 14px;
                --ag-font-family: 'Source Sans Pro', sans-serif;
            }
            .feature-column { background-color: rgba(46, 139, 87, 0.2) !important; }
            .target-column { background-color: rgba(255, 99, 71, 0.2) !important; }
        """
    }

    # Configuration des colonnes
    for col in dataset.data.columns:
        column_type = ("🎯 Variable cible" if col in dataset.target_columns else
                       "✨ Variable explicative" if col in dataset.features_columns else
                       "Variable non utilisée")

        cell_class = ("target-column" if col in dataset.target_columns else
                      "feature-column" if col in dataset.features_columns else "")

        gb.configure_column(
            col,
            headerTooltip=f"{column_type}\n\nType: {dataset.data[col].dtype}",
            cellClass=cell_class
        )

    # Configuration globale de la grille
    gb.configure_pagination(enabled=True, paginationAutoPageSize=False, paginationPageSize=15)
    gb.configure_default_column(sortable=True, filterable=True)

    grid_options = gb.build()

    return AgGrid(
        dataset.data,
        gridOptions=grid_options,
        custom_css=grid_style,
        height=500,
        theme='streamlit',
        fit_columns_on_grid_load=True,
        update_mode=GridUpdateMode.MODEL_CHANGED
    )


def display_variable_selection(dataset):
    """Affiche et gère la sélection des variables."""
    st.markdown("## 🎯 Configuration des variables")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Variables explicatives")
        features = st.multiselect(
            "Sélectionnez les variables explicatives",
            options=[col for col in dataset.data.columns if col not in dataset.target_columns],
            default=dataset.features_columns,
            help="Ces variables seront utilisées pour faire les prédictions"
        )

        if features:
            st.info(f"📊 {len(features)} variables explicatives sélectionnées")

    with col2:
        st.markdown("### Variables cibles")
        targets = st.multiselect(
            "Sélectionnez les variables cibles",
            options=[col for col in dataset.data.columns if col not in features],
            default=dataset.target_columns,
            help="Ces variables sont celles que vous souhaitez prédire"
        )

        if targets:
            st.info(f"🎯 {len(targets)} variables cibles sélectionnées")

    # Validation et mise à jour
    if (set(features) != set(dataset.features_columns) or
            set(targets) != set(dataset.target_columns)):
        with st.expander("⚠️ Modifications non sauvegardées"):
            st.warning(
                "Vous avez modifié la sélection des variables. "
                "N'oubliez pas de sauvegarder vos changements."
            )
            if st.button("💾 Sauvegarder les modifications", type="primary"):
                dataset.features_columns = features
                dataset.target_columns = targets
                DatasetState.set_dataset(dataset)
                st.success("✅ Configuration mise à jour avec succès!")
                st.rerun()


def main():
    """Point d'entrée principal de la page."""
    dataset = DatasetState.get_dataset()

    if not dataset:
        with st.container():
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown("### :ghost: Aucun jeu de données")
                st.caption("Chargez un jeu de données pour commencer l'analyse")
                st.button(
                    "📤 Charger un jeu de données",
                    type="primary",
                    on_click=dataset_config_form,
                    use_container_width=True
                )
        return

    # Affichage organisé des informations
    display_dataset_overview(dataset)

    # Tabs pour organiser le contenu
    tab1, tab2, tab3 = st.tabs([
        "📊 Aperçu des données",
        "📈 Qualité des données",
        "⚙️ Configuration"
    ])

    with tab1:
        configure_grid_display(dataset)

    with tab2:
        display_data_quality_insights(dataset)

    with tab3:
        display_variable_selection(dataset)


main()