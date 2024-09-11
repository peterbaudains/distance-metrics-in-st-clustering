import os
from neo4j import GraphDatabase
from dotenv import load_dotenv
import pandas as pd
import geopandas as gpd
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))

from clustering.euclidean_dbscan import euclideanDBSCAN
from clustering.frame_split_method import frame_split_method
from data_loader.neo4j_data_loader import DataLoaderNeo4j

import numpy as np
load_dotenv()
import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger()

def get_driver():
    # Returns neo4j db driver
    uri=os.environ['NEO4J_SERVER']
    driver=GraphDatabase.driver(uri, auth=(os.environ['NEO4J_USER'], os.environ['NEO4J_PASSWORD']))
    return driver

minTime = '2023-11-08T09:40:10'
maxTime = '2023-11-08T09:54:00'
extent =  [-0.117613, -0.113493, 51.504565, 51.509373]
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

cluster_algo = euclideanDBSCAN(d_eps=d_eps, t_eps=t_eps, min_samples=min_samples)
merged_labels = frame_split_method(gdf, cluster_algo, frame_size=25000)

gdf['cluster'] = merged_labels
gdf['recordedAtTime'] = pd.to_datetime(gdf['unix_time'], unit='s')

res = gdf[gdf['cluster'] > 0].groupby('cluster')\
                        .agg({'recordedAtTime': ["min", "max", np.size], 
                                'longitude': ["mean"], 
                                'latitude': ["mean"], 
                                'vehicleRef': ['nunique']})

gdf.explore()