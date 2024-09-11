import pandas as pd
import geopandas as gpd
import logging
import time
import numpy as np
from shapely import Point

from clustering.dbscan import DBSCAN
from clustering.frame_split_method import frame_split_method
log = logging.getLogger("experiment")

maxSpeed = 0.3

def run_experiment(df: pd.DataFrame, cluster_algo: DBSCAN, frame_size: int, exp_reference: str) -> None:
    
    df_slow = df[df['speed'] < maxSpeed].copy()

    log.info("Number of records for clustering: %s" % df_slow.shape[0])

    df_slow['unix_time'] = ((df_slow[['recordedAtTime']] - \
                    pd.Timestamp("1970-01-01")) // \
                    pd.Timedelta('1s'))['recordedAtTime'].values
    
    df_slow = df_slow[['unix_time', 'lon2', 'lat2', 'vehicleRef', 'vehicleJourneyRef', 'directionRef', 'lineRef']]
    df_slow = df_slow.sort_values(by='unix_time')
    df_slow.columns = ['unix_time', 'longitude', 'latitude', 'vehicleRef', 'vehicleJourneyRef', 'directionRef', 'lineRef']
    df_slow.reset_index(drop=True, inplace=True)
    
    df['points'] = df.apply(lambda x: Point(x.lon2, x.lat2), axis=1)
    df['unix_time'] = ((df[['recordedAtTime']] - pd.Timestamp("1970-01-01")) // \
                        pd.Timedelta('1s'))['recordedAtTime'].values

    gdf = gpd.GeoDataFrame(df_slow, 
                           geometry=gpd.points_from_xy(df_slow['longitude'], df_slow['latitude']), 
                           crs=4326)
    
    t1_split = time.time()
    log.info('Running frame split method for experiment %s' % exp_reference)
    merged_labels = frame_split_method(gdf, cluster_algo, frame_size=frame_size)
    t2_split = time.time()

    log.info(f"Time taken for clustering: {(t2_split - t1_split):.2f}")

    gdf['cluster'] = merged_labels
    gdf['recordedAtTime'] = pd.to_datetime(gdf['unix_time'], unit='s')

    filename = 'outputs/%s.csv' % exp_reference
    gdf[gdf['cluster'] > 0].groupby('cluster')\
                           .agg({'recordedAtTime': ["min", "max", np.size], 
                                 'longitude': ["mean"], 
                                 'latitude': ["mean"], 
                                 'vehicleRef': ['nunique']})\
                           .to_csv(filename, index=False)