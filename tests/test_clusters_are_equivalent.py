
def get_cluster_label_remapper(list_of_cluster_labels: list) -> dict:
    cluster_remapper = {-1:-1}
    max_cluster = 0
    for c in list_of_cluster_labels:
        if c not in cluster_remapper.keys():
            cluster_remapper[c] = max_cluster + 1
            max_cluster += 1
    return cluster_remapper

def test_cluster_labels_are_equivalent(a: dict, b:dict) -> None:

    assert [k for k,v in a.items() if v > -1] == [k for k,v in b.items() if v > -1]

    a_clusters =  [v for v in a.values() if v > -1]
    b_clusters = [v for v in b.values() if v > -1]

    assert len(a_clusters) == len(b_clusters)

    a_remapper = get_cluster_label_remapper(list(a.values()))
    a_remapped = {k: a_remapper[c] for k, c in a.items()}

    b_remapper = get_cluster_label_remapper(list(b.values()))
    b_remapped = {k: b_remapper[c] for k,c in b.items()}
    #assert a_remapped == b_remapped
    print({k: i==j for k,i,j in zip(a_remapped.keys(), a_remapped.values(), b_remapped.values()) if i!=j})
