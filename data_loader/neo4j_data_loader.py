import os
import time
import logging
import sys
import pandas as pd
import cartopy.crs as ccrs
from neo4j import GraphDatabase, time as neo4jtime
from dotenv import load_dotenv

# Load dotenv
load_dotenv()

# Configure logging
log = logging.getLogger(__name__)

class DataLoaderNeo4j:

    def __init__(self):
        self.uri = os.environ['NEO4J_SERVER']
        self.auth = (os.environ['NEO4J_USER'], os.environ['NEO4J_PASSWORD'])

    def get_driver(self):
        # Returns neo4j db driver
        uri=self.uri
        driver=GraphDatabase.driver(uri, auth=self.auth)
        return driver

    def get_obs_for_time_period(self, tx, extent: list, minTime: str, maxTime: str):
        # Returns the source data for the viz, for a particular date
        # and with a geographic extent

        min_lon, max_lon, min_lat, max_lat = extent

        CYPHER_QUERY = f"""
        MATCH (n1:Observation)-[s:SAME_JOURNEY]->(n2:Observation)
        WHERE n2.recordedAtTime >= datetime($minTime)
        AND n2.recordedAtTime < datetime($maxTime)
        AND n2.geometry.x >= $min_lon
        AND n2.geometry.x < $max_lon
        AND n2.geometry.y >= $min_lat
        AND n2.geometry.y < $max_lat
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
        return tx.run(CYPHER_QUERY, 
                    minTime=minTime, 
                    maxTime=maxTime,
                    min_lon=min_lon, 
                    max_lon=max_lon, 
                    min_lat=min_lat, 
                    max_lat=max_lat).to_df()

    def load_df(self, extent, minTime: str, maxTime: str):
        t = time.time()
        log.info('Loading data')
        # Get data
        driver = self.get_driver()
        with driver.session(database=os.environ['DB_NAME']) as session:
            df = session.execute_read(self.get_obs_for_time_period, 
                                      extent=extent, 
                                      minTime=minTime, 
                                      maxTime=maxTime)
        
        log.info('Data loaded from neo4j')
        # Convert recordedAtTime to datetime
        df['recordedAtTime'] = df['recordedAtTime'].apply(neo4jtime.DateTime.to_native).dt.tz_localize(None)
        return df
