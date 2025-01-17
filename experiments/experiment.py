import pandas as pd
import geopandas as gpd
import logging
import time
import numpy as np
from shapely import Point

from clustering.dbscan import DBSCAN
from clustering.frame_split_method import frame_split_method
from clustering.cluster_uncertainty import calculate_cluster_certainty
from clustering.post_processor import merge_occluding_clusters
from clustering.network_dbscan import networkDBSCAN
from clustering.euclidean_dbscan import euclideanDBSCAN
log = logging.getLogger("experiment")


def run_experiment(df: pd.DataFrame, cluster_algo: DBSCAN, max_speed: float,
                   frame_size: int, exp_reference: str, save_obs: bool, simplify=True) -> None:
    
    df_slow = df[df['speed'] < max_speed].copy()

    log.info("Number of records for clustering: %s" % df_slow.shape[0])

    # Calculate unix time from recordedAtTime
    df_slow['unix_time'] = ((df_slow[['recordedAtTime']] - \
                    pd.Timestamp("1970-01-01")) // \
                    pd.Timedelta('1s'))['recordedAtTime'].values
    
    df_slow = df_slow[['unix_time', 'lon2', 'lat2', 'vehicleRef', 'vehicleJourneyRef', 'directionRef', 'lineRef']]
    df_slow = df_slow.sort_values(by='unix_time')
    df_slow_column_names = ['unix_time', 'longitude', 'latitude', 'vehicleRef', 'vehicleJourneyRef', 'directionRef', 'lineRef']
    df_slow.columns = df_slow_column_names
    df_slow.reset_index(inplace=True)

    df['unix_time'] = ((df[['recordedAtTime']] - pd.Timestamp("1970-01-01")) // \
                        pd.Timedelta('1s'))['recordedAtTime'].values

    gdf = gpd.GeoDataFrame(df_slow, 
                           geometry=gpd.points_from_xy(df_slow['longitude'], df_slow['latitude']), 
                           crs=4326)
    
    t1_split = time.time()
    log.info('Running frame split method for experiment %s' % exp_reference)
    merged_labels = frame_split_method(gdf[df_slow_column_names + ['geometry']], cluster_algo, frame_size=frame_size)
    t2_split = time.time()

    log.info(f"Time taken for clustering: {(t2_split - t1_split):.2f}")

    gdf['cluster'] = merged_labels
    gdf['recordedAtTime'] = pd.to_datetime(gdf['unix_time'], unit='s')

    # Add the cluster column back to the full dataset for subsequent analysis.
    df = df.merge(gdf[['index', 'cluster']], 
                  left_index=True, 
                  right_on='index', 
                  how='left')
    if save_obs:
        # Save the result.
        obs_out = f'outputs/obs_{exp_reference}.csv'
        df.to_csv(obs_out, index=False)

    filename = 'outputs/%s.csv' % exp_reference
    cluster_df = gdf[gdf['cluster'] > 0].groupby('cluster')\
                           .agg({'recordedAtTime': ["min", "max", np.size], 
                                 'longitude': ["mean"], 
                                 'latitude': ["mean"], 
                                 'vehicleRef': ["nunique"],
                                 'unix_time': ["min", "max"]})

    if cluster_df.shape[0] > 1:
        cluster_df = merge_occluding_clusters(cluster_df, df)

    cluster_df.columns = ['recordedAtTimeMin', 'recordedAtTimeMax', 'nObs', \
                        'longitude', 'latitude', 'nVehicleUnique', \
                        'unixTimeMin', 'unixTimeMax']
    
    # Add a cluster column, which is needed for the cluster certainty 
    # calculation in neo4j and added here to ensure consistency of the output
    # files.
    cluster_df.reset_index(drop=False, names='cluster', inplace=True)

    # We don't need this for the NRT runs but we do need it for the two week runs...
    if type(cluster_algo)==networkDBSCAN:
        calculation_type = 'network'
    elif type(cluster_algo)==euclideanDBSCAN:
        calculation_type = 'eucl'
    if cluster_df.shape[0] > 0:
        cluster_df = calculate_cluster_certainty(cluster_df, 
                                                calculation_type=calculation_type,
                                                distance_buffer=200, 
                                                speed_threshold=max_speed, 
                                                simplify=simplify)

    cluster_df.to_csv(filename, index=False, index_label='cluster')