def data2nodes(data, pipe, graph):
    label2node = {v: k for k, v in enumerate(zip(graph.vs['pullback_set_label'], graph.vs['partial_cluster_label']))}
    pullback_cover = pipe.named_steps['pullback_cover'].transform(data)
    clustering = pipe.named_steps['clustering'].transform(pullback_cover)

    nodelist = []
    for labels in clustering:
        label = []
        for lab in labels:
            if lab in list(label2node.keys()):
                label.append(label2node[lab])
        label = label if len(label) > 0 else -1
        nodelist.append(label)
    return nodelist
