import pandas as pd
import os
from neo4j import GraphDatabase
import osmnx as ox
from clustering.network_dbscan import node_query, rels_query, next_closest_intersection_query, insert_data, execute_query
import logging

log = logging.getLogger()

def get_driver():
    uri=os.environ['NEO4J_SERVER']
    driver=GraphDatabase.driver(uri, auth=(os.environ['NEO4J_USER'],os.environ['NEO4J_PASSWORD']))
    return driver

closest_intersection_query = """
UNWIND $rows AS row
CREATE (o:Observation {id: row.index, unix_time: row.unix_time, speed:row.speed_ms, geometry: point({srid:4326, x:row.longitude, y: row.latitude})})
WITH row, o
MATCH (v:Intersection {osmid: row.nearest_node})
CREATE (o)-[:CLOSEST_INTERSECTION {length: row.distance}]->(v), (v)-[:CLOSEST_INTERSECTION {length: row.distance}]->(o)
RETURN COUNT(*) AS total
"""

closest_intersection_query_centroid = """
UNWIND $rows AS row
CREATE (o:Centroid {id: row.cluster, unix_time_min: row.unixTimeMin, unix_time_max:row.unixTimeMax, geometry: point({srid:4326, x:row.longitude, y: row.latitude})})
WITH row, o
MATCH (v:Intersection {osmid: row.nearest_node})
CREATE (o)-[:CLOSEST_INTERSECTION {length: row.distance}]->(v), (v)-[:CLOSEST_INTERSECTION {length: row.distance}]->(o)
RETURN COUNT(*) AS total
"""

project_graph_query = """
CALL gds.graph.project(
    'network_distance',               
    ['Intersection', 'Observation', 'Centroid'],
    ['CLOSEST_INTERSECTION', 'ROAD_SEGMENT'],
    {relationshipProperties:'length'}
);"""

def get_cluster_certainty(tx, d_eps, speed_threshold):
    get_neighbourhood_data = """
        MATCH (source: Centroid), (target: Observation)
        WHERE point.distance(source.geometry, target.geometry) < $d_eps
        AND source <> target 
        AND target.unix_time >= source.unix_time_min
        AND target.unix_time < source.unix_time_max
        CALL gds.shortestPath.dijkstra.stream('network_distance', {
            sourceNode:source, 
            targetNode:target, 
            relationshipWeightProperty: 'length'
        })
        YIELD sourceNode, targetNode, totalCost
        WITH source, target, sourceNode, targetNode, totalCost
        WHERE totalCost < $d_eps
        RETURN 
            gds.util.asNode(sourceNode).id as sourceNodeId, 
            sum(case when gds.util.asNode(targetNode).speed < $speed_threshold then 1 else 0 end) as slow_moving_observations, 
            sum(case when gds.util.asNode(targetNode).speed >= $speed_threshold then 1 else 0 end) as fast_moving_observations, 
            toFloat(sum(case when gds.util.asNode(targetNode).speed < $speed_threshold then 1 else 0 end)) / COUNT(*) as cluster_certainty
        """
    return tx.run(get_neighbourhood_data, d_eps=d_eps, speed_threshold=speed_threshold).to_df()


def get_nearby_data_for_network_cluster_certainty_calc(tx, s, d_eps):
    CYPHER_QUERY = """
    MATCH (o1:Observation)-[r:SAME_JOURNEY]->(o2:Observation)
    WHERE o1.recordedAtTime >= datetime($recordedAtTimeMin)
    AND o1.recordedAtTime < datetime($recordedAtTimeMax)
    AND point.distance(o1.geometry, point({srid:4326, x:$lng, y:$lat})) < $d_eps
    RETURN o1.geometry.x as longitude, o1.geometry.y as latitude, o1.recordedAtTime, r.speed_ms as speed_ms
    """
    return tx.run(CYPHER_QUERY, 
                  recordedAtTimeMin=s['recordedAtTimeMin'].strftime('%Y-%m-%dT%H:%M:%SZ'), 
                  recordedAtTimeMax=s['recordedAtTimeMax'].strftime('%Y-%m-%dT%H:%M:%SZ'), 
                  lng=s['longitude'], lat=s['latitude'], d_eps=d_eps).to_df()


def get_neighbourhood_data_by_cluster(cluster_df, distance_buffer):              
    df = pd.DataFrame()
    driver=get_driver()
    with driver.session(database='busopendata') as session:
        for i, s in cluster_df.iterrows():
            df_sub = session.execute_read(get_nearby_data_for_network_cluster_certainty_calc, s, distance_buffer)
            df_sub['cluster_index'] = i
            df = pd.concat([df, df_sub])
    if df.shape[0] == 0:
        raise ValueError("No data populated in df.")
    df['o1.recordedAtTime'] = df['o1.recordedAtTime'].apply(lambda x: x.to_native()).dt.tz_localize(None)
    df['unix_time'] = ((df[['o1.recordedAtTime']] - pd.Timestamp("1970-01-01")) // \
                        pd.Timedelta('1s'))['o1.recordedAtTime'].values

    return df


def calculate_cluster_certainty(cluster_df: pd.DataFrame, 
                                calculation_type: str,
                                distance_buffer:float, 
                                speed_threshold: float, 
                                simplify:bool) -> pd.DataFrame:
    '''

    Parameters
    ----------
    cluster_df is a dataframe which is saved as the output of the 
    run_experiment function in experiment.py

    '''
    df_nbhd = get_neighbourhood_data_by_cluster(cluster_df, distance_buffer)

    if calculation_type == 'eucl':
        slow = df_nbhd[df_nbhd['speed_ms'] < speed_threshold].groupby('cluster_index').size()
        fast = df_nbhd[df_nbhd['speed_ms'] >= speed_threshold].groupby('cluster_index').size()
        slow.name='slow_moving_observations'
        fast.name='fast_moving_observations'
        certainty_df = pd.concat([slow, fast], axis=1)
        certainty_df['cluster_certainty'] = certainty_df['slow_moving_observations'] / (certainty_df['slow_moving_observations'] + certainty_df['fast_moving_observations'])
        df = cluster_df.merge(certainty_df, left_index=True, right_index=True)

    if calculation_type == 'network':
        extent = [-0.16172376,-0.07189224,51.49288835,51.52433822]
        extent_reformatted = [extent[3], extent[2], extent[0], extent[1]]
        G = ox.graph_from_bbox(bbox=extent_reformatted, network_type='drive', simplify=simplify)
        gdf_nodes, gdf_relationships = ox.graph_to_gdfs(G)
        gdf_nodes.reset_index(inplace=True)
        gdf_relationships.reset_index(inplace=True)
        
        driver=get_driver()
        with driver.session(database="networkdistancetest") as session:
            session.execute_write(execute_query, "MATCH (n) DETACH DELETE n")
            session.execute_write(execute_query, "CALL gds.graph.drop('network_distance',false)")
            session.execute_write(insert_data, node_query, gdf_nodes.drop(columns=['geometry']))
            session.execute_write(insert_data, rels_query, gdf_relationships.drop(columns=['geometry']))

        df_nbhd['nearest_node'], df_nbhd['distance'] = ox.nearest_nodes(G, df_nbhd['longitude'], df_nbhd['latitude'], return_dist=True)
        cluster_df['nearest_node'], cluster_df['distance'] = ox.nearest_nodes(G, cluster_df['longitude'], cluster_df['latitude'], return_dist=True)

        with driver.session(database="networkdistancetest") as session:
            session.execute_write(insert_data, closest_intersection_query, df_nbhd.reset_index())
            session.execute_write(insert_data, closest_intersection_query_centroid, cluster_df.reset_index())
            session.execute_write(execute_query, next_closest_intersection_query)
            session.execute_write(execute_query, project_graph_query)
            neighbourhood_data = session.execute_read(get_cluster_certainty, distance_buffer, speed_threshold)
            session.execute_write(execute_query, "CALL gds.graph.drop('network_distance',false)")
            session.execute_write(execute_query, "MATCH (o:Observation) DETACH DELETE o")
        df = cluster_df.merge(neighbourhood_data, left_index=True, right_on='sourceNodeId')
    return df