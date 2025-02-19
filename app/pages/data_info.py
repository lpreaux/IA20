# app/pages/data_info.py
import streamlit as st
from dataset.state import DatasetState
from dataset.forms.config import dataset_config_form
from st_aggrid import AgGrid, GridOptionsBuilder, DataReturnMode, GridUpdateMode


def render_no_dataset_skeleton():
    """Affiche un squelette de la page quand aucun dataset n'est chargé."""
    with st.container():
        # En-tête
        st.markdown("# Jeu de données")

        # Skeleton pour le message d'erreur
        with st.container(border=True):
            # Utilisation des colonnes pour centrer le contenu
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown("### :ghost: Aucun jeu de données")
                st.caption("Chargez un jeu de données pour commencer l'analyse")
                st.button("📤 Charger un jeu de données",
                          type="primary",
                          on_click=dataset_config_form,
                          use_container_width=True)

        # Skeleton pour le tableau
        with st.container():
            st.markdown("##### Aperçu des données")
            st.dataframe({"": []}, use_container_width=True)

        # Skeleton pour les colonnes
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Variables explicatives")
            st.dataframe({"": []}, use_container_width=True)
        with col2:
            st.markdown("##### Variables cibles")
            st.dataframe({"": []}, use_container_width=True)


def get_grid_height(dataset, rows_per_page=25):
    """Calcule la hauteur dynamique du grid en fonction du nombre de lignes."""
    # Hauteur minimale
    min_height = 400
    # Hauteur par ligne (incluant padding)
    row_height = 35
    # Hauteur de l'en-tête
    header_height = 52
    # Hauteur de la pagination
    pagination_height = 40

    # Calcul du nombre de lignes à afficher
    num_rows = min(rows_per_page, len(dataset.data))

    # Calcul de la hauteur totale
    total_height = (num_rows * row_height) + header_height + pagination_height

    return max(min_height, total_height)


def render_column_selection():
    """Affiche et permet la modification des colonnes sélectionnées."""
    dataset = DatasetState.get_dataset()

    # Aperçu des données avec AgGrid
    st.markdown("### Aperçu des données")

    # Configuration de AgGrid
    gb = GridOptionsBuilder.from_dataframe(dataset.data)

    # Style des en-têtes pour améliorer la lisibilité
    header_style = {
        "cssText": """
            .ag-header-cell-label {
                justify-content: center;
                font-family: 'Source Sans Pro', sans-serif;
                font-size: 14px;
                white-space: normal !important;
                line-height: 1.2;
                padding: 5px;
            }
            .ag-header-cell {
                background-color: #1E1E1E;
                min-height: 60px;
            }
            .ag-theme-streamlit {
                --ag-header-height: 60px;
                --ag-header-foreground-color: #FFFFFF;
                --ag-header-background-color: #1E1E1E;
                --ag-odd-row-background-color: #0E1117;
                --ag-row-hover-color: rgba(255, 255, 255, 0.1);
                --ag-selected-row-background-color: rgba(255, 255, 255, 0.2);
                --ag-font-size: 14px;
                --ag-font-family: 'Source Sans Pro', sans-serif;
            }
        """
    }

    # Configuration des colonnes
    for col in dataset.data.columns:
        tooltip = ("✨ Feature" if col in dataset.features_columns
                   else "🎯 Target" if col in dataset.target_columns
        else None)

        cell_style = None
        if col in dataset.features_columns:
            cell_style = {'backgroundColor': 'rgba(38, 77, 31, 0.3)'}
        elif col in dataset.target_columns:
            cell_style = {'backgroundColor': 'rgba(77, 31, 31, 0.3)'}

        gb.configure_column(
            col,
            headerTooltip=tooltip,
            wrapHeaderText=True,
            autoHeaderHeight=True,
            wrapText=True,
            cellStyle=cell_style,
            minWidth=100
        )

    # Configuration de la pagination
    rows_per_page = 20
    gb.configure_pagination(
        enabled=True,
        paginationAutoPageSize=False,
        paginationPageSize=rows_per_page
    )

    # Configuration globale
    gb.configure_default_column(
        resizable=True,
        sortable=True,
        filterable=True,
        wrapText=True
    )
    gb.configure_grid_options(
        domLayout='normal',
        headerHeight=60,
        rowHeight=35,
        suppressRowHoverHighlight=False,
        enableRangeSelection=True
    )

    gridOptions = gb.build()

    # Calcul de la hauteur dynamique
    grid_height = get_grid_height(dataset, rows_per_page)

    # Affichage du grid
    grid_response = AgGrid(
        dataset.data,
        gridOptions=gridOptions,
        height=grid_height,
        custom_css=header_style,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        update_mode=GridUpdateMode.SELECTION_CHANGED | GridUpdateMode.VALUE_CHANGED,
        theme='streamlit'
    )

    # Sélection des colonnes
    st.markdown("### Sélection des colonnes")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Variables explicatives")
        features = st.multiselect(
            "Sélectionnez les variables explicatives",
            options=[col for col in dataset.data.columns
                     if col not in dataset.target_columns],
            default=dataset.features_columns,
            key="features_selector"
        )

    with col2:
        st.markdown("##### Variables cibles")
        targets = st.multiselect(
            "Sélectionnez les variables cibles",
            options=[col for col in dataset.data.columns
                     if col not in features],
            default=dataset.target_columns,
            key="targets_selector"
        )

    # Bouton de mise à jour
    if (set(features) != set(dataset.features_columns) or
            set(targets) != set(dataset.target_columns)):
        if st.button("💾 Mettre à jour la sélection", type="primary"):
            # Mise à jour du dataset
            dataset.features_columns = features
            dataset.target_columns = targets
            DatasetState.set_dataset(dataset)
            st.success("✅ Sélection mise à jour")
            st.rerun()


# Point d'entrée de la page
dataset = DatasetState.get_dataset()

if not dataset:
    render_no_dataset_skeleton()
else:
    st.markdown("# Jeu de données")
    render_column_selection()
