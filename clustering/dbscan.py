from abc import abstractmethod
import geopandas as gpd
import logging

log = logging.getLogger("DBSCAN")

class DBSCAN:

    def __init__(self, d_eps, t_eps, min_samples):
        self.d_eps = d_eps
        self.t_eps = t_eps
        self.min_samples = min_samples
    
    def fit(self, data: gpd.GeoDataFrame) -> None:
        self.set_data(data)
        self.labels = [0] * len(data)
        self.core = [0] * len(data)
        cluster_label = 0

        for i in range(len(data)):
            if i % 100 == 0:
                log.debug('Progress complete: %s' % (i / len(data)))
            if self.labels[i] != 0:
                continue
            neighbours = self._retrieve_neighbours(i)
            if len(neighbours) + 1 < self.min_samples: # we add one since we don't return the current observation.
                self.labels[i] = -1
            else:
                cluster_label += 1
                log.debug('Setting %s as first obs in cluster %s' % (i, cluster_label))
                self.labels[i] = cluster_label
                self.core[i] = 1
                log.debug('Expanding cluster %s' % cluster_label)
                self._expand_cluster(i, neighbours, cluster_label)

    def _expand_cluster(self, i:int, neighbours:list, cluster_label: int) -> None:
        for neighbour in neighbours:
            if self.labels[neighbour] == -1:
                log.debug('Adding %s to cluster %s' % (neighbour, cluster_label))
                self.labels[neighbour] = cluster_label
            elif self.labels[neighbour] == 0:
                log.debug('Adding %s to cluster %s' % (neighbour, cluster_label))
                self.labels[neighbour] = cluster_label
                new_neighbours = self._retrieve_neighbours(neighbour)
                if len(new_neighbours) + 1 >= self.min_samples:
                    self.core[neighbour] = 1
                    neighbours += new_neighbours

    @abstractmethod
    def set_data(self, data: gpd.GeoDataFrame) -> None:
        pass

    @abstractmethod
    def _retrieve_neighbours(self, i: int) -> list:
        pass
