from clustering.dbscan import DBSCAN
import geopandas as gpd
import numpy as np
import logging

log = logging.getLogger()

class euclideanDBSCAN(DBSCAN):

    def __init__(self, d_eps, t_eps, min_samples):
        DBSCAN.__init__(self, d_eps, t_eps, min_samples)

    def set_data(self, data: gpd.GeoDataFrame) -> None:
        # Convert to projected coordinate system 
        data = data.to_crs(27700)
        self.data = data
    
    def _retrieve_neighbours(self, i):
        log.debug('Retrieving neighbours for index %s' % i)
        
        neighbours = self.data[
            (self.data.geometry.distance(self.data.iloc[i].geometry) < self.d_eps) & 
            (np.abs(self.data.unix_time - self.data.iloc[i].unix_time) <= self.t_eps)
            ]

        if neighbours.shape[0] > 0:
            return neighbours.drop(self.data.iloc[i].name).index.values.tolist()
        else:
            return []