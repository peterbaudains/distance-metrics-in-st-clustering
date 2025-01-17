
import sys
import os
from neo4j import GraphDatabase
from os.path import dirname, realpath
from dotenv import load_dotenv
sys.path.append(dirname(dirname(realpath(__file__))))

load_dotenv()

from clustering.network_dbscan import networkDBSCAN
from data_loader.neo4j_data_loader import DataLoaderNeo4j
from experiments.experiment import run_experiment

import datetime as dt
import logging
import pandas as pd
import time
log = logging.getLogger(__name__)

def get_driver():
    uri=os.environ['NEO4J_SERVER']
    driver=GraphDatabase.driver(uri, auth=(os.environ['NEO4J_USER'], 
                                           os.environ['NEO4J_PASSWORD']))
    return driver

if __name__ == "__main__":
    print('Running')
    date = dt.datetime.now()
    date_str = date.strftime("%Y%m%d")
    logging.basicConfig(filename="logs/net_experiment_%s.log" % date_str, 
                        filemode='a', level=logging.INFO)
    
    extent = [-0.16172376,-0.07189224,51.49288835,51.52433822]
    start_tw = "2023-11-01"
    end_tw = "2023-11-15"
    maxSpeed = 0.3
    d_eps = 50
    t_eps = 300
    min_samples = 10
    time_index = pd.date_range(start_tw, end_tw, freq='15min')


    for simplify in [True, False]:

        # First run, which is done separately here to ensure we build the correct 
        # street network for the remaining experiments. 
        log.info(f"Starting experiment for time range {str(time_index[0] - dt.timedelta(0, 7200))} - {str(time_index[0])}")
        cluster_algo = networkDBSCAN(d_eps=d_eps, t_eps=t_eps, 
                                    min_samples=min_samples, extent=extent, 
                                    neo4jdriver=get_driver(), simplify=simplify, 
                                    reload_sn=True)
        
        minTime = str(time_index[0] - dt.timedelta(0, 7200)).replace(' ', 'T')
        maxTime = str(time_index[0]).replace(' ', 'T')
        
        df = DataLoaderNeo4j().load_df(extent=extent, 
                                    minTime=minTime, 
                                    maxTime=maxTime)

        run_experiment(df, 
                       cluster_algo, 
                       max_speed=maxSpeed, 
                       frame_size=7200, 
                       exp_reference=f'nrt_network_twoweeks_simplify{simplify}_d{d_eps}\\{date_str}_nrt_net_t{t_eps}_d{d_eps}+ending{maxTime.replace(':','-')}',
                       save_obs=False)

        cluster_algo = networkDBSCAN(d_eps=d_eps, t_eps=t_eps, 
                                    min_samples=min_samples, extent=extent, 
                                    neo4jdriver=get_driver(), simplify=simplify, 
                                    reload_sn=False)

        for ti in time_index[1:]:        
            log.info(f"Starting experiment for time range {str(ti - dt.timedelta(0, 7200))} - {str(ti)}")
            t1 = time.time()
            maxTime = str(ti).replace(' ', 'T')
            minTime = str(ti - dt.timedelta(0, 7200)).replace(' ', 'T')
            df = DataLoaderNeo4j().load_df(extent=extent, 
                                        minTime=minTime, 
                                        maxTime=maxTime)

            run_experiment(df, 
                        cluster_algo, 
                        max_speed=maxSpeed, 
                        frame_size=7200, 
                        exp_reference=f'nrt_network_twoweeks_simplify{simplify}_d{d_eps}\\{date_str}_nrt_net_t{t_eps}_d{d_eps}+ending{maxTime.replace(':','-')}',
                        save_obs=False)
            
            t2 = time.time()
            log.info(f'Time taken: {t2 - t1}')