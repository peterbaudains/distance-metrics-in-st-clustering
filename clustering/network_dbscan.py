from neo4j import Neo4jDriver
from clustering.dbscan import DBSCAN

import geopandas as gpd
import osmnx as ox
import logging

log = logging.getLogger()

closest_intersection_query = """
UNWIND $rows AS row
CREATE (o:Observation {id: row.index, unix_time: row.unix_time, geometry: point({srid:4326, x:row.longitude, y: row.latitude})})
WITH row, o
MATCH (v:Intersection {osmid: row.nearest_node})
CREATE (o)-[:CLOSEST_INTERSECTION {length: row.distance}]->(v), (v)-[:CLOSEST_INTERSECTION {length: row.distance}]->(o)
RETURN COUNT(*) AS total
"""

node_query = """
UNWIND $rows AS row
WITH row WHERE row.osmid IS NOT NULL
MERGE (i:Intersection {osmid: row.osmid})
    SET i.location = point({latitude: row.y, longitude: row.x}), 
        i.ref = row.ref,
        i.highway = row.highway, 
        i.street_count = toInteger(row.street_count)
RETURN count(*) as total
"""

rels_query = """
UNWIND $rows AS road
MATCH (u:Intersection {osmid: road.u})
MATCH (v:Intersection {osmid: road.v})
MERGE (u)-[r:ROAD_SEGMENT {osmid: road.osmid}]->(v)
SET r.oneway = road.oneway,
r.lanes = road.lanes,
r.ref = road.ref,
r.name = road.name,
r.highway = road.highway,
r.max_speed = road.maxspeed,
r.length = toFloat(road.length)
RETURN COUNT(*) AS total
"""

next_closest_intersection_query = """
match (o)-[:CLOSEST_INTERSECTION]->(i:Intersection)-[:ROAD_SEGMENT]->(i2:Intersection)
with o, i2, point.distance(o.geometry, i2.location) as distance
order by o.id, distance
with o, apoc.agg.first(i2) as next_accessible_closest_intersection
CREATE (o)-[:CLOSEST_INTERSECTION {length: point.distance(o.geometry, next_accessible_closest_intersection.location)}]->(next_accessible_closest_intersection), (next_accessible_closest_intersection)-[:CLOSEST_INTERSECTION {length: point.distance(o.geometry, next_accessible_closest_intersection.location)}]->(o)
return count(*) as total
"""

project_graph_query = """
CALL gds.graph.project(
    'network_distance',               
    ['Intersection', 'Observation'],
    ['CLOSEST_INTERSECTION', 'ROAD_SEGMENT'],
    {relationshipProperties:'length'}
);"""

def get_neighbourhood_data(tx, d_eps, t_eps):
    get_neighbourhood_data = """
        MATCH (source: Observation), (target: Observation)
        WHERE point.distance(source.geometry, target.geometry) < $d_eps 
        AND source <> target 
        AND abs(source.unix_time - target.unix_time) < $t_eps
        CALL gds.shortestPath.dijkstra.stream('network_distance', {
            sourceNode:source, 
            targetNode:target, 
            relationshipWeightProperty: 'length'
        })
        YIELD sourceNode, targetNode, totalCost
        WITH source, target, sourceNode, targetNode, totalCost
        where totalCost < $d_eps
        RETURN 
            gds.util.asNode(sourceNode).id as sourceNodeId, 
            gds.util.asNode(targetNode).id as targetNodeId, 
            totalCost, 
            point.distance(source.geometry, target.geometry)
        """
    return tx.run(get_neighbourhood_data, d_eps=d_eps, t_eps=t_eps).to_df()



def insert_data(tx, query, rows, batch_size=25000):
    total = 0
    batch = 0
    while batch * batch_size < len(rows):
        results = tx.run(query, parameters = {
                    'rows': rows[batch * batch_size: (batch + 1) * batch_size]
                    .to_dict('records')
                    }).data()
        print(batch * batch_size, results)
        total += results[0]['total']
        batch += 1

def execute_query(tx, query):
    tx.run(query)

class networkDBSCAN(DBSCAN):

    def __init__(self, d_eps, t_eps, min_samples, extent, neo4jdriver, simplify=True):
        DBSCAN.__init__(self, d_eps, t_eps, min_samples)
        # Expects a fresh db
        self.extent = extent
        self.driver = neo4jdriver
        self.simplify=simplify

        # This is always called as the first thing in fit.
        # So we can load the data into a fresh db, 
        # project the graph and prepare for the shortest path search.
        self.G = self.get_graph_from_osmnx(self.extent)
        
        # Prepare data to load into neo4j
        gdf_nodes, gdf_relationships = ox.graph_to_gdfs(self.G)
        gdf_nodes.reset_index(inplace=True)
        gdf_relationships.reset_index(inplace=True)

        with self.driver.session(database="networkdistancetest") as session:
            session.execute_write(execute_query, "MATCH (n) DETACH DELETE n")
            session.execute_write(execute_query, "CALL gds.graph.drop('network_distance',false)")
            session.execute_write(insert_data, node_query, gdf_nodes.drop(columns=['geometry']))
            session.execute_write(insert_data, rels_query, gdf_relationships.drop(columns=['geometry']))
    
    def set_data(self, data: gpd.GeoDataFrame) -> None:
        self.data = data
        
        # Find nearest nodes using osmnx
        self.data['nearest_node'], self.data['distance'] = \
            ox.nearest_nodes(self.G, self.data['longitude'], self.data['latitude'], 
                             return_dist=True)
        
        with self.driver.session(database="networkdistancetest") as session:
            session.execute_write(insert_data, closest_intersection_query, self.data.drop(columns=['geometry']).reset_index())
            session.execute_write(execute_query, next_closest_intersection_query)
            session.execute_write(execute_query, project_graph_query)
            self.neighbourhood_data = session.execute_read(get_neighbourhood_data, self.d_eps, self.t_eps)
            session.execute_write(execute_query, "CALL gds.graph.drop('network_distance',false)")
            session.execute_write(execute_query, "MATCH (o:Observation) DETACH DELETE o")

    def get_graph_from_osmnx(self, extent:list):
        # Reformat for osmnx spec
        extent_reformatted = [extent[3], extent[2], extent[0], extent[1]]
        return ox.graph_from_bbox(bbox=extent_reformatted, network_type='drive', simplify=self.simplify)
        
    def _retrieve_neighbours(self, i):
        log.debug('Retrieving neighbours for index %s' % i)
        neighbours = self.neighbourhood_data[self.neighbourhood_data['sourceNodeId']==i]['targetNodeId'].tolist()
        return neighbours



if __name__=="__main__": 

    # Run a test
    
    from shapely import Point
    import numpy as np

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
    
    # In format north, south, east, west
    extent = [51.52433821827542, 51.49288835442542, -0.1617237642059759, -0.07189223579402408]
    test_extent = [51.523205,51.497564,-0.174908,-0.09972]

    d_eps = 50
    t_eps = 300
    min_samples = 10
    cluster_algo = networkDBSCAN(d_eps=d_eps, t_eps=t_eps, min_samples=min_samples, extent=test_extent)
    cluster_algo.fit(data=gdf)    

