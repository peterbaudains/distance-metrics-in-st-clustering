import sys
import os
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))

from shapely import Point
import numpy as np
import geopandas as gpd
from clustering.network_dbscan import networkDBSCAN
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

def get_driver():
    uri=os.environ['NEO4J_SERVER']
    driver=GraphDatabase.driver(uri, auth=(os.environ['NEO4J_USER'],os.environ['NEO4J_PASSWORD']))
    return driver

sample_points = np.array([
    Point(-0.122688, 51.510017),
    Point(-0.120623, 51.510878),
    Point(-0.118924, 51.512239),
    Point(-0.119168, 51.51174),
    Point(-0.119096, 51.511044),
    Point(-0.11986, 51.516439),
    Point(-0.116133, 51.507777),
    Point(-0.11986, 51.516439),
    Point(-0.11986, 51.516439),
    Point(-0.118158, 51.517761),
    Point(-0.119972, 51.51766),
    Point(-0.124799, 51.517628),
    Point(-0.128343, 51.510204),
    Point(-0.127829, 51.509686),
    Point(-0.127284, 51.509148),
    Point(-0.128086, 51.509971), 
    Point(-0.128086, 51.509971),
    Point(-0.127509,51.507213),
    Point(-0.128086, 51.509971)
])

sample_times = [i * 100 for i in range(len(sample_points))]

gdf = gpd.GeoDataFrame({'unix_time':sample_times, 
                        'latitude':[p.y for p in sample_points], 
                        'longitude':[p.x for p in sample_points]}, 
                        geometry=sample_points, 
                        crs=4326)

test_extent = [-0.174908,-0.09972,51.497564,51.523205]

driver = get_driver()
d_eps = 50
t_eps = 300
min_samples = 10
cluster_algo = networkDBSCAN(d_eps=d_eps, 
                             t_eps=t_eps, 
                             min_samples=min_samples, 
                             extent=test_extent, 
                             neo4jdriver=driver, 
                             simplify=True)
cluster_algo.fit(data=gdf)
