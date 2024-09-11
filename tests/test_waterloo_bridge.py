import os
from neo4j import GraphDatabase
from dotenv import load_dotenv
import pandas as pd
import geopandas as gpd
import sys
from clustering.network_dbscan import networkDBSCAN
from clustering.euclidean_dbscan import euclideanDBSCAN
from clustering.frame_split_method import frame_split_method
import numpy as np
load_dotenv()
import logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
log = logging.getLogger()

def get_driver():
    # Returns neo4j db driver
    uri=os.environ['NEO4J_SERVER']
    driver=GraphDatabase.driver(uri, auth=(os.environ['NEO4J_USER'], os.environ['NEO4J_PASSWORD']))
    return driver

def get_waterloo_bridge_obs(tx):
    CYPHER_QUERY = """MATCH (n1:Observation)-[s:SAME_JOURNEY]->(n2:Observation)
    WHERE n2.recordedAtTime >= '2023-11-08 09:40:10'
    AND n2.recordedAtTime < '2023-11-08 09:54:00'
    AND n2.geometry.x >= -0.117613
    AND n2.geometry.x < -0.113493
    AND n2.geometry.y >= 51.504565
    AND n2.geometry.y < 51.509373
    RETURN n2.recordedAtTime as recordedAtTime, 
        n2.vehicleRef as vehicleRef, 
        n2.vehicleJourneyRef as vehicleJourneyRef, 
        n2.directionRef as directionRef, 
        n2.lineRef as lineRef,
        n1.geometry.x as lon1, 
        n1.geometry.y as lat1,
        n2.geometry.x as lon2, 
        n2.geometry.y as lat2, 
        n2.itemIdentifier as itemIdentifier, 
        s.speed_ms as speed"""
    return tx.run(CYPHER_QUERY).to_df()


extent = [-0.117613, -0.113493, 51.504565, 51.509373]
driver = get_driver()
with driver.session(database=os.environ['DB_NAME']) as session:
    df = session.execute_read(get_waterloo_bridge_obs)
df['recordedAtTime'] = pd.to_datetime(df['recordedAtTime']).dt.tz_localize(None)
df_slow = df[df['speed'] < 0.3]
df_slow['unix_time'] = ((df_slow[['recordedAtTime']] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s'))['recordedAtTime'].values
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