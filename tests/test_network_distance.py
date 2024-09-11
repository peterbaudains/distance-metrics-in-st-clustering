import logging
import pandas as pd
import geopandas as gpd
import time
import numpy as np

import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))

from clustering.network_dbscan import networkDBSCAN
from clustering.frame_split_method import frame_split_method
from data_loader.neo4j_data_loader import DataLoaderNeo4j
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

load_dotenv()

def get_driver():
    uri=os.environ['NEO4J_SERVER']
    driver=GraphDatabase.driver(uri, auth=(os.environ['NEO4J_USER'],os.environ['NEO4J_PASSWORD']))
    return driver

log = logging.getLogger(__name__)

if __name__=="__main__":

    logging.basicConfig(level=logging.INFO)

    neo4j_logger = logging.getLogger('neo4j')
    neo4j_logger.setLevel('INFO')

    extent = [-0.16172376,-0.07189224,51.49288835,51.52433822]
    
    minTime = "2023-11-08T06:30:00"
    maxTime = "2023-11-08T14:00:00"
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
    driver = get_driver()

    t1_split = time.time()
    cluster_algo = networkDBSCAN(d_eps=d_eps, t_eps=t_eps, min_samples=min_samples, extent=extent, neo4jdriver=driver)
    merged_labels = frame_split_method(gdf, cluster_algo, frame_size=25000)
    t2_split = time.time()
    log.info('Time taken: %.2f' % (t2_split - t1_split))


    gdf['cluster'] = merged_labels
    gdf['recordedAtTime'] = pd.to_datetime(gdf['unix_time'], unit='s')

    gdf[gdf['cluster'] > 0].groupby('cluster')\
                           .agg({'recordedAtTime': ["min", "max", np.size], 
                                 'longitude': ["mean"], 
                                 'latitude': ["mean"], 
                                 'vehicleRef': ['nunique']})\
                           .to_csv('outputs\\test_network_distance.csv', index=False)