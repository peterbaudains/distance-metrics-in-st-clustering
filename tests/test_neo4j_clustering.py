import logging
import time
import numpy as np
import pandas as pd
import geopandas as gpd
from st_clustering import ST_DBSCAN

import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))

from clustering.frame_split_method import frame_split_method
from clustering.euclidean_dbscan import euclideanDBSCAN
from data_loader.neo4j_data_loader import DataLoaderNeo4j
from tests.test_clusters_are_equivalent import test_cluster_labels_are_equivalent

log = logging.getLogger(__name__)

if __name__=="__main__":

    from shapely import Point    
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    extent = [-0.16172376,-0.07189224,51.49288835,51.52433822]
    minTime = "2023-11-08T09:00:00"
    maxTime = "2023-11-08T12:00:00"
    maxSpeed = 0.3

    df = DataLoaderNeo4j().load_df(extent=extent, minTime=minTime, 
                                   maxTime=maxTime)
    df_slow = df[df['speed'] < maxSpeed].copy()

    log.info("Number of records for clustering: %s" % df_slow.shape[0])
    
    df_slow['unix_time'] = ((df_slow[['recordedAtTime']] - \
                    pd.Timestamp("1970-01-01")) // \
                    pd.Timedelta('1s'))['recordedAtTime'].values
    
    df_slow = df_slow[['unix_time', 'lon2', 'lat2', 'vehicleRef', 'vehicleJourneyRef', 'directionRef', 'lineRef']]
    df_slow = df_slow.sort_values(by='unix_time')
    df_slow.columns = ['unix_time', 'longitude', 'latitude', 'vehicleRef', 'vehicleJourneyRef', 'directionRef', 'lineRef']
    df_slow.reset_index(drop=True, inplace=True)

    gdf = gpd.GeoDataFrame(df_slow, 
                           geometry=gpd.points_from_xy(df_slow['longitude'], df_slow['latitude']), 
                           crs=4326)

    d_eps = 50
    t_eps = 300
    min_samples = 10

    t1_split = time.time()
    cluster_algo = euclideanDBSCAN(d_eps=d_eps, t_eps=t_eps, min_samples=min_samples-2)
    merged_labels = frame_split_method(gdf, cluster_algo)
    t2_split = time.time()

    t1_old = time.time()
    cluster_algo = euclideanDBSCAN(d_eps=d_eps, t_eps=t_eps, min_samples=min_samples-1)
    cluster_algo.fit(data=gdf)
    t2_old = time.time()
    
    t1_lib = time.time()
    gdf = gdf.to_crs(27700)
    unix_times = gdf['unix_time'].values.tolist()
    eastings = [i.x for i in gdf.geometry]
    northings = [i.y for i in gdf.geometry]
    cluster_vals = np.stack((unix_times, eastings, northings), axis=1)
    hdb = ST_DBSCAN(eps1=d_eps, eps2=t_eps, min_samples=min_samples)
    hdb.st_fit(cluster_vals)
    hdb_labels = hdb.labels
    hdb_labels = [(i + (1 * (i>=0))) for i in hdb_labels]
    t2_lib = time.time()

    log.info(f"Time taken for split method: {(t2_split - t1_split):.2f}")
    log.info(f"Time taken for original method: {(t2_old - t1_old):.2f}")
    log.info(f"Time taken for library method: {(t2_lib - t1_lib):.2f}")
    print({k:v for k, v in zip(gdf.index, cluster_algo.labels) if v > -1})
    print({k:v for k, v in merged_labels.items() if v > -1})
    print({k:v for k, v in zip(gdf.index, hdb_labels) if v > -1})

    #test_cluster_labels_are_equivalent({k:v for k, v in zip(gdf.index, cluster_algo.labels)}, merged_labels)
    #test_cluster_labels_are_equivalent({k:v for k, v in zip(gdf.index, hdb_labels)}, merged_labels)
    #test_cluster_labels_are_equivalent({k:v for k, v in zip(gdf.index, hdb_labels)}, {k:v for k, v in zip(gdf.index, cluster_algo.labels)})
