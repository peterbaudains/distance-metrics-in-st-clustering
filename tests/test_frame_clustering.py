import numpy as np  
import geopandas as gpd
from shapely import Point

import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))

from frame_split_method import frame_split_method
from euclidean_dbscan import euclideanDBSCAN

if __name__=="__main__":

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

    iso_eps = 1./3.
    t_eps = 200
    d_eps = 300
    min_samples = 3
    cluster_algo = euclideanDBSCAN(d_eps, t_eps=t_eps, min_samples=min_samples)
    merged_labels = frame_split_method(gdf, cluster_algo)
    cluster_algo.fit(gdf)
    full_frame_labels = cluster_algo.labels
    print(merged_labels)
    print(dict(enumerate(full_frame_labels)))
    
    assert merged_labels == dict(enumerate(full_frame_labels)), "labels not matching"

    # Test against st_clustering package
    from st_clustering import ST_DBSCAN
    gdf = gdf.to_crs(27700)
    unix_times = gdf['unix_time'].values.tolist()
    eastings = [i.x for i in gdf.geometry]
    northings = [i.y for i in gdf.geometry]
    cluster_vals = np.stack((unix_times, eastings, northings), axis=1)
    hdb = ST_DBSCAN(eps1=d_eps, eps2=t_eps, min_samples=min_samples)
    hdb.st_fit(cluster_vals)
    hdb_labels = hdb.labels

    hdb_labels = [(i + (1 * (i>=0))) for i in hdb_labels]

    print(dict(enumerate(hdb_labels)))