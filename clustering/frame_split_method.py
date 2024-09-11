from lib2to3.pgen2.literals import test
from clustering.dbscan import DBSCAN
import geopandas as gpd
import logging 
import pandas as pd
import numpy as np

log = logging.getLogger()

def common_core_point(merged: pd.DataFrame) -> bool:
    sum_cores = merged[['core_x', 'core_y']].sum(axis=1, keepdims=False)
    sum_cores_is_two = (sum_cores == 2)
    return bool(np.any(sum_cores_is_two, keepdims=False))


def core_plus_marginal_point(merged: pd.DataFrame) -> bool:
    marginal_or_sum_cores = merged[(merged['cluster_y']>-1) & \
                                   (merged[['core_x', 'core_y']]\
                                    .sum(axis=1, keepdims=False) >= 1)]
    return bool(np.any(marginal_or_sum_cores, keepdims=False))


def test_for_merging_previous_clusters(merged: pd.DataFrame) -> list:
    df = (merged.groupby('cluster_y').apply(core_plus_marginal_point) & \
            merged.groupby('cluster_y').apply(common_core_point))
    if (df.shape[0] > 0 and df.sum() > 1):
        return df.loc[df].index.values.tolist()
    else:
        return []


def implement_cluster_matching(prev: pd.DataFrame, curr: pd.DataFrame, 
                               labels: dict, prev_cluster_map: dict) \
                                -> tuple[dict, dict]:

    clusters_to_merge = {}
    current_frame_cluster_lookup = {}

    # We need a check here to see if two distinct clusters from the previous
    # dataframe need merging as a result of the clustering from curr.

    # We iterate through the clusters in curr, join on the observations in 
    # previous (for each cluster in curr) and determine whether there are 
    # at least two distinct clusters in prev which share core points or at 
    # least one core point and a marginal point, as per Peca (2012).
    prev_clusters_to_merge = []
    for cluster in curr['cluster'].unique():
        if cluster == -1:
            continue

        cluster_list = []
        merged = curr[curr['cluster']==cluster].merge(
                    prev[['cluster', 'core', 'original_index']],
                    on='original_index', 
                    how='inner')[['core_x', 'core_y', 'cluster_x', 'cluster_y', 'original_index']]

        # Remove noise
        merged = merged[merged['cluster_y'] > -1]

        # Implement merging test and store any clusters that need merging.
        prev_clusters_to_merge.append(test_for_merging_previous_clusters(merged))

    for cluster_list in prev_clusters_to_merge:
        if len(cluster_list) > 1:
            global_cluster_labels_to_merge = [prev_cluster_map[i] for i in cluster_list]
            new_cluster_label = min(global_cluster_labels_to_merge)
            indexes_to_update = [i for i,v in labels.items() if v in global_cluster_labels_to_merge]
            for index in indexes_to_update:
                labels[index] = new_cluster_label
            for index in cluster_list:
                prev_cluster_map[index] = new_cluster_label

    # This loop is to match clusters in prev and curr
    # which need merging - the actual re-labelling process is performed in the 
    # subsequent loop.
    for prev_cluster_frame, prev_cluster_overall in prev_cluster_map.items():

        merged = prev[prev['cluster'] == prev_cluster_frame].merge(
                    curr[['cluster', 'core', 'original_index']], 
                    on='original_index', 
                    how='inner')

        if common_core_point(merged) or core_plus_marginal_point(merged):  
            # Then these clusters will need merging
            for cluster in merged['cluster_y'].unique():
                if cluster > -1:
                    clusters_to_merge[cluster] = prev_cluster_overall
    
    # Now we loop through and re-label if necessary.
    for i, row in curr.iterrows():
        if row['original_index'] in labels.keys() and labels[row['original_index']] > -1:
            log.debug('Not doing anything for obs %s. Current frame cluster: %s, overall cluster: %s' % \
                      (row['original_index'], row['cluster'], labels[row['original_index']]))
            continue
        elif row['cluster'] in clusters_to_merge.keys():
            # This obs has been linked to a cluster that needs merging. 
            # So this cluster must already exist in labels - we need to get this cluster id. 
            labels[row['original_index']] = clusters_to_merge[row['cluster']]
            current_frame_cluster_lookup[row['cluster']] = clusters_to_merge[row['cluster']]
            log.debug('Relabelling obs %s to overall cluster %s from frame cluster %s'  % \
                      (row['original_index'], labels[row['original_index']], row['cluster']))
        elif row['cluster'] in current_frame_cluster_lookup.keys():
            labels[row['original_index']] = current_frame_cluster_lookup[row['cluster']]
            log.debug('Relabelling obs %s to overall cluster %s from frame cluster %s' % \
                      (row['original_index'], labels[row['original_index']], row['cluster']))
        elif row['cluster'] > -1:
            # Then it's a new cluster we haven't seen before. Record the label and map it to a new value.
            current_frame_cluster_lookup[row['cluster']] = max(1, max(labels.values()) + 1)
            labels[row['original_index']] = current_frame_cluster_lookup[row['cluster']]
            log.debug('Adding new overall cluster %s for observation %s and frame cluster %s' % \
                      (labels[row['original_index']], row['original_index'], row['cluster']))
        else:
            labels[row['original_index']] = -1
            log.debug('Setting obs %s to noise.' % (row['original_index']))

    return labels, current_frame_cluster_lookup


def frame_split_method(gdf: gpd.GeoDataFrame,  
                       cluster_algo: DBSCAN, 
                       frame_size=None) -> dict:
    
    # Set parameter defaults
    if frame_size is None:
        frame_size = 4 * cluster_algo.t_eps
    frame_overlap = 2 * cluster_algo.t_eps

    # Initialise previous frame variable
    prev_gdf_frame = None

    # Loop through frames, using 'unix_time' column to define time.
    for i in range(gdf['unix_time'].min(), gdf['unix_time'].max(), (frame_size - frame_overlap + 1)):
        
        if i > gdf['unix_time'].min():
            # Update prev_gdf_frame
            prev_gdf_frame = gdf_frame[['original_index', 'cluster', 'core']].copy()
        
        # Current frame definition
        gdf_frame = gdf[(gdf['unix_time'] >= i) & \
                        (gdf['unix_time'] <= frame_size + i)]\
                        .copy()\
                        .reset_index(names='original_index')
        
        log.info(f"Frame size: {gdf_frame.shape[0]}")
        log.info(f"Min index: {gdf_frame['original_index'].min()}")
        log.info(f"Max index: {gdf_frame['original_index'].max()}")

        # Fit to frame
        cluster_algo.fit(gdf_frame)
        gdf_frame['cluster'] = cluster_algo.labels
        gdf_frame['core'] = cluster_algo.core

        if prev_gdf_frame is not None: 
            merged_labels, prev_cluster_map = implement_cluster_matching(prev_gdf_frame, gdf_frame, merged_labels, prev_cluster_map)
        else:
            merged_labels = dict(zip(gdf_frame.original_index.values, gdf_frame.cluster.values))
            prev_cluster_map = {k:k for k in np.unique(gdf_frame.cluster.values) if k > 0}

    # Do it one more time at the end
    merged_labels, prev_cluster_map = implement_cluster_matching(gdf_frame[['original_index', 'cluster', 'core']].copy(), 
                                                                 pd.DataFrame(columns=['original_index', 'cluster', 'core']), 
                                                                 merged_labels, 
                                                                 prev_cluster_map)

    return merged_labels
