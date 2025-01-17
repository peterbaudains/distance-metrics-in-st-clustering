'''
This script reads in a cluster output file and generates a new cluster file
which contains glyphs which do not overlap.
'''

import pandas as pd
import geopandas as gpd
from scipy.spatial.distance import pdist, squareform
import numpy as np


def get_closest_overlapping_cluster(gdf, spatial_threshold):

    overlaps_mask = np.empty(shape=[gdf.shape[0],gdf.shape[0]])
    overlaps_mask[:] = np.nan

    for i, l in enumerate(gdf['time_overlaps']):
        for j in l:
            if i != j:
                overlaps_mask[i][j] = 1

    dist_ij = overlaps_mask * squareform(
                pdist(np.stack([gdf.geometry.x, gdf.geometry.y], axis=1))
              )

    try: 
        ind = np.unravel_index(np.nanargmin(dist_ij, axis=None), dist_ij.shape)
    except ValueError:
        return []
    
    min_dist = dist_ij[ind]

    if min_dist < spatial_threshold:
        return ind
    else:
        return []

 
def merge_occluding_clusters(clusters: pd.DataFrame, obs: pd.DataFrame):

    '''
    This function merges clusters that are too close together in space and time
    and which will therefore suffer from occlusion - i.e. where one cluster is 
    plotted on top of another. 

    The clusters df provided will have the following columns:    
                recordedAtTime                          longitude latitude vehicleRef unix_time    
                            min                 max size      mean     mean    nunique       min max
    cluster
    1       2024-12-01 09:00:00 2024-12-01 09:30:00    2       1.5    41.45          2         0   1
    2       2024-12-01 10:00:00 2024-12-01 10:00:00    1       3.0    41.60          1         2   2
    3       2024-12-01 10:15:00 2024-12-01 10:30:00    2       4.5    41.95          1         3   4
    4       2024-12-01 10:40:00 2024-12-01 10:40:00    1       6.0    43.20          1         5   5

    The obs df will contains a row per observation and is needed here to 
    re-calculate the number of unique vehicles. 
    '''

    gdf = gpd.GeoDataFrame(clusters, 
                           geometry=gpd.points_from_xy(
                               clusters[('longitude', 'mean')], 
                               clusters[('latitude', 'mean')]), 
                           crs=4326)
    
    # We'll be calculating distances so need to project this gdf
    gdf = gdf.to_crs(27700)

    # Calculate the radius of the glyph in the visualisation in metres. 
    # For points that are within this radius with an overlapping time period
    
    # In matplotlib.pyplot.scatter, 
    # the marker size in points**2 (typographic points are 1/72 in.). 

    # our size value to vizent add_glyphs is 40
    # Then s=(size)**2

    # So the 40 value corresponds to points (or 1/72 inches * 40)

    # 1 point == fig.dpi / 72. pixels
    # fig.dpi = 172
    # So 1 point = 172 / 72 pixels
    # So 40 points = (172 / 72) * 40 pixels = 95.5 pixels

    # We are plotting 3840 x 2160 pixels.

    # Covering an area of 10000 x 5625 in web-mercator units. 
    
    # For the web-mercator projection, this corresponds to an area of 
    # 6236m x 3657m

    # So 3840 / 6236 = 0.616 pixels per metre.
    # And 2160 / 3657 = 0.590 pixels per metre.

    # Which means that 95.5 pixels corresponds to 95.5 / 0.616 = 155
    gdf['interval'] = gdf.apply(
                        lambda x: pd.Interval(
                                    x[('unix_time', 'min')], 
                                    x[('unix_time', 'max')], 
                                    closed='both'), 
                                axis=1)

    gdf['time_overlaps'] = gdf['interval'].apply(
            lambda x: np.asarray([x.overlaps(y) for y in gdf.interval]).nonzero()[0]
        )

    spatial_threshold = 160
    gdf = gdf.reset_index(drop=True)
    cluster_to_merge = get_closest_overlapping_cluster(gdf, spatial_threshold)
    
    while len(cluster_to_merge) > 0:
        print('cluster to merge: ', cluster_to_merge)
        new_cluster_label = gdf.index.max() + 1
        
        gdf.loc[new_cluster_label, ('recordedAtTime', 'min')] = gdf.loc[cluster_to_merge, ('recordedAtTime', 'min')].min()
        gdf.loc[new_cluster_label, ('recordedAtTime', 'max')] = gdf.loc[cluster_to_merge, ('recordedAtTime', 'max')].max()
        gdf.loc[new_cluster_label, ('recordedAtTime', 'size')] = gdf.loc[cluster_to_merge, ('recordedAtTime', 'size')].sum()
        gdf.loc[new_cluster_label, ('longitude', 'mean')] = (gdf.loc[cluster_to_merge, ('recordedAtTime', 'size')] * gdf.loc[cluster_to_merge, ('longitude', 'mean')]).sum() / gdf.loc[cluster_to_merge, ('recordedAtTime', 'size')].sum()
        gdf.loc[new_cluster_label, ('latitude', 'mean')] = (gdf.loc[cluster_to_merge, ('recordedAtTime', 'size')] * gdf.loc[cluster_to_merge, ('latitude', 'mean')]).sum() / gdf.loc[cluster_to_merge, ('recordedAtTime', 'size')].sum()
        gdf.loc[new_cluster_label, ('unix_time', 'min')] = gdf.loc[cluster_to_merge, ('unix_time', 'min')].min()
        gdf.loc[new_cluster_label, ('unix_time', 'max')] = gdf.loc[cluster_to_merge, ('unix_time', 'max')].max()
        gdf.loc[new_cluster_label, 'interval'] = pd.Interval(gdf.loc[new_cluster_label, ('unix_time', 'min')], 
                                                             gdf.loc[new_cluster_label, ('unix_time', 'max')], 
                                                             closed='both')
        gdf.loc[new_cluster_label, 'geometry'] = gpd.GeoSeries(gpd.points_from_xy([gdf.loc[new_cluster_label, ('longitude', 'mean')]],[gdf.loc[new_cluster_label, ('latitude', 'mean')]]), crs=4326).to_crs(27700).values[0]   
        gdf.loc[new_cluster_label, ('vehicleRef', 'nunique')] = obs[obs['cluster'].isin(cluster_to_merge)].vehicleRef.nunique()

        for row_to_drop in cluster_to_merge:
            print('dropping row: ', row_to_drop)

            gdf.drop(row_to_drop, axis=0, inplace=True)
        
        gdf['time_overlaps'] = gdf['interval'].apply(
            lambda x: np.asarray([x.overlaps(y) for y in gdf.interval]).nonzero()[0]
        )
        gdf = gdf.reset_index(drop=True)
        cluster_to_merge = get_closest_overlapping_cluster(gdf, spatial_threshold)

    
    return gdf[[('recordedAtTime', 'min'), ('recordedAtTime', 'max'), 
                ('recordedAtTime', 'size'), ('longitude', 'mean'), 
                ('latitude', 'mean'), ('vehicleRef', 'nunique'), 
                ('unix_time', 'min'), ('unix_time', 'max')]]
