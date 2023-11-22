from sklearn.pipeline import Pipeline


from gtda.mapper.nerve import Nerve
from gtda.mapper.utils._list_feature_union import ListFeatureUnion
from gtda.mapper.utils.pipeline import transformer_from_callable_on_rows, identity
from gtda.mapper.pipeline import MapperPipeline

from .cluster import PredictiveParallelClustering

global_pipeline_params = ("memory", "verbose")
nodes_params = ("scaler", "filter_func", "cover")
clust_prepr_params = ("clustering_preprocessing",)
clust_params = ("clusterer", "n_jobs",
                "parallel_backend_prefer")
nerve_params = ("min_intersection", "store_edge_elements", "contract_nodes")
clust_prepr_params_prefix = "pullback_cover__"
nodes_params_prefix = "pullback_cover__map_and_cover__"
clust_params_prefix = "clustering__"
nerve_params_prefix = "nerve__"


def make_predictive_mapper_pipeline(scaler=None,
                                    filter_func=None,
                                    cover=None,
                                    clustering_preprocessing=None,
                                    clusterer=None,
                                    n_jobs=None,
                                    parallel_backend_prefer=None,
                                    graph_step=True,
                                    min_intersection=1,
                                    store_edge_elements=False,
                                    contract_nodes=False,
                                    memory=None,
                                    verbose=False):

    if scaler is None:
        _scaler = identity(validate=False)
    else:
        _scaler = scaler

    # If filter_func is not a scikit-learn transformer, assume it is a callable
    # to be applied on each row separately. Then attempt to create a
    # FunctionTransformer object to implement this behaviour.
    if filter_func is None:
        from sklearn.decomposition import PCA
        _filter_func = PCA(n_components=2)
    elif not hasattr(filter_func, "fit_transform"):
        _filter_func = transformer_from_callable_on_rows(filter_func)
    else:
        _filter_func = filter_func

    if cover is None:
        from gtda.mapper.cover import CubicalCover
        _cover = CubicalCover()
    else:
        _cover = cover

    if clustering_preprocessing is None:
        _clustering_preprocessing = identity(validate=True)
    else:
        _clustering_preprocessing = clustering_preprocessing

    if clusterer is None:
        from sklearn.cluster import DBSCAN
        _clusterer = DBSCAN()
    else:
        _clusterer = clusterer

    map_and_cover = Pipeline(
        steps=[("scaler", _scaler),
               ("filter_func", _filter_func),
               ("cover", _cover)],
        verbose=verbose)

    all_steps = [
        ("pullback_cover", ListFeatureUnion(
            [("clustering_preprocessing", _clustering_preprocessing),
             ("map_and_cover", map_and_cover)])),
        ("clustering", PredictiveParallelClustering(
            _clusterer,
            n_jobs=n_jobs,
            parallel_backend_prefer=parallel_backend_prefer))
        ]

    if graph_step:
        all_steps.append(
            ("nerve", Nerve(min_intersection=min_intersection,
                            store_edge_elements=store_edge_elements,
                            contract_nodes=contract_nodes))
            )

    mapper_pipeline = MapperPipeline(
        steps=all_steps, memory=memory, verbose=verbose)

    return mapper_pipeline
