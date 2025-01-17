import sys
import time
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
    
    extent = [-0.16172376,-0.07189224,51.49288835,51.52433822]
    minTime = "2023-11-01"
    maxTime = "2023-11-15"
    maxSpeed = 0.3
    d_eps = 50
    t_eps = 300
    min_samples = 10

    t1 = time.time()
    df = DataLoaderNeo4j().load_df(extent=extent, minTime=minTime, maxTime=maxTime)
    t2 = time.time()
    log.info(f'Data loaded. Time taken: {t2 - t1}')

    driver = get_driver()
 
    for simplify in [False]:

        cluster_algo = networkDBSCAN(d_eps=d_eps, 
                                     t_eps=t_eps, 
                                     min_samples=min_samples, 
                                     extent=extent, 
                                     neo4jdriver=driver, 
                                     simplify=simplify, 
                                     reload_sn=True)
        
        run_experiment(df, cluster_algo, max_speed=maxSpeed, frame_size=10800, 
                       exp_reference=f'{date_str}_network_simplify{simplify}_1Nov23to15Nov23_d{d_eps}_t{t_eps}', 
                       save_obs=False, simplify=simplify)
    
        log.info(f'Experiment complete. Total time taken: {time.time() - t1}')
        