from gtda.mapper.utils._visualization import (
    _validate_color_kwargs,
    _calculate_graph_data,
    _produce_static_figure,
)


def plot_static_graph(graph, data, color_data=None, color_features=None,
                      node_color_statistic=None, layout="kamada_kawai", layout_dim=2,
                      n_sig_figs=3, node_scale=12, plotly_params=None):
    (color_data_transformed, column_names_dropdown,
     node_color_statistic) = \
        _validate_color_kwargs(graph, data, color_data, color_features,
                               node_color_statistic, interactive=False)

    edge_trace, node_trace, node_colors_color_features = \
        _calculate_graph_data(
            graph, color_data_transformed, node_color_statistic, layout,
            layout_dim, n_sig_figs, node_scale
            )

    figure = _produce_static_figure(
        edge_trace, node_trace, node_colors_color_features,
        column_names_dropdown, layout_dim, n_sig_figs, plotly_params
        )
    return figure
