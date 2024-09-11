
import sys
import os
from neo4j import GraphDatabase
from os.path import dirname, realpath
from dotenv import load_dotenv
sys.path.append(dirname(dirname(realpath(__file__))))

load_dotenv()

from clustering.network_dbscan import networkDBSCAN
from data_loader.neo4j_data_loader import DataLoaderNeo4j
from experiment import run_experiment

import datetime as dt
import logging
import pandas as pd
log = logging.getLogger(__name__)

def get_driver():
    uri=os.environ['NEO4J_SERVER']
    driver=GraphDatabase.driver(uri, auth=(os.environ['NEO4J_USER'],os.environ['NEO4J_PASSWORD']))
    return driver

if __name__ == "__main__":

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

    cluster_algo = networkDBSCAN(d_eps=d_eps, t_eps=t_eps, min_samples=min_samples, extent=extent, neo4jdriver=get_driver(), simplify=True)

    for ti in pd.date_range(start_tw, end_tw, freq='15min'):        
        maxTime = str(ti).replace(' ', 'T')
        minTime = str(ti - dt.timedelta(0, 7200)).replace(' ', 'T')

        df = DataLoaderNeo4j().load_df(extent=extent, minTime=minTime, maxTime=maxTime)
    
        run_experiment(df, cluster_algo, frame_size=7200, 
                    exp_reference='nrt_network_run\\twoweeks_simplify_d%s\\%s_nrt_net_t%s_d%s+ending%s' % \
                        (d_eps, date_str, t_eps, d_eps, maxTime.replace(' ','_').replace(':','-')))