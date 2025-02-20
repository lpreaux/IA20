
def get_color_palette(n_classes):
    """Génère une palette de couleurs cohérente pour n classes.

    Parameters:
    -----------
    n_classes : int
        Nombre de classes uniques

    Returns:
    --------
    dict
        Dictionnaire associant chaque classe à une couleur
    """
    # Palette de couleurs Plotly par défaut étendue
    colors = [
        '#1f77b4',  # bleu
        '#ff7f0e',  # orange
        '#2ca02c',  # vert
        '#d62728',  # rouge
        '#9467bd',  # violet
        '#8c564b',  # marron
        '#e377c2',  # rose
        '#7f7f7f',  # gris
        '#bcbd22',  # jaune-vert
        '#17becf',  # cyan
        '#aec7e8',  # bleu clair
        '#ffbb78',  # orange clair
        '#98df8a',  # vert clair
        '#ff9896',  # rouge clair
        '#c5b0d5',  # violet clair
    ]

    # Si plus de couleurs sont nécessaires, on utilise une interpolation
    if n_classes > len(colors):
        from plotly.colors import n_colors
        colors = n_colors('rgb(5, 200, 200)', 'rgb(200, 10, 10)', n_classes, colortype='rgb')

    return colors[:n_classes]