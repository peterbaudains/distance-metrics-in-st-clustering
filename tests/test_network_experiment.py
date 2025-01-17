import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))

from clustering.network_dbscan import networkDBSCAN
from data_loader.neo4j_data_loader import DataLoaderNeo4j
from experiments.experiment import run_experiment

import datetime as dt
import logging
log = logging.getLogger(__name__)
import os
from neo4j import GraphDatabase

def get_driver():
    uri=os.environ['NEO4J_SERVER']
    driver=GraphDatabase.driver(uri, auth=(os.environ['NEO4J_USER'],os.environ['NEO4J_PASSWORD']))
    return driver

if __name__ == "__main__":

    date = dt.datetime.now()
    date_str = date.strftime("%Y%m%d")
    logging.basicConfig(filename="logs/network_experiment_%s.log" % date_str, 
                        filemode='a', level=logging.INFO)
    
    # Experimental parameters
    extent = [-0.16172376,-0.07189224,51.49288835,51.52433822]
    minTime = "2023-11-08T06:30:00"
    maxTime = "2023-11-08T14:00:00"
    maxSpeed = 0.3
    d_eps = 50
    t_eps = 300
    min_samples = 10

    df = DataLoaderNeo4j().load_df(extent=extent, 
                                   minTime=minTime, 
                                   maxTime=maxTime)
    log.info(f"Data loaded: {df.shape[0]} records from Neo4j")
    
    # Need a new driver to pass to the clustering algorithm
    driver = get_driver()

    # Build the clustering algorithm
    cluster_algo = networkDBSCAN(d_eps=d_eps, 
                                 t_eps=t_eps, 
                                 min_samples=min_samples, 
                                 extent=extent, 
                                 neo4jdriver=driver, 
                                 simplify=True, 
                                 reload_sn=True)
    
    # Run the experiment
    run_experiment(df=df, 
                   cluster_algo=cluster_algo,    
                   max_speed=maxSpeed,           
                   frame_size=10800, 
                   exp_reference=f'{date_str}_network_test_d{d_eps}_t{t_eps}', 
                   save_obs=True, 
                   simplify=True)
    