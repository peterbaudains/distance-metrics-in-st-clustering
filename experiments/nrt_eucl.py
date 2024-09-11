
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))

from clustering.euclidean_dbscan import euclideanDBSCAN
from data_loader.neo4j_data_loader import DataLoaderNeo4j
from experiment import run_experiment

import datetime as dt
import logging
import pandas as pd
log = logging.getLogger(__name__)

if __name__ == "__main__":

    date = dt.datetime.now()
    date_str = date.strftime("%Y%m%d")
    logging.basicConfig(filename="logs/eucl_experiment_%s.log" % date_str, 
                        filemode='a', level=logging.INFO)
    
    extent = [-0.16172376,-0.07189224,51.49288835,51.52433822]
    start_tw = "2023-11-01"
    end_tw = "2023-11-15"
    maxSpeed = 0.3
    t_eps = 300
    min_samples = 10
    d_eps = 25
    cluster_algo = euclideanDBSCAN(d_eps=d_eps, t_eps=t_eps, min_samples=min_samples)

    for ti in pd.date_range(start_tw, end_tw, freq='15min'):                    
        maxTime = str(ti)
        minTime = str(ti - dt.timedelta(0, 7200))
        df = DataLoaderNeo4j().load_df(extent=extent, minTime=minTime, maxTime=maxTime)
        run_experiment(df, cluster_algo, frame_size=7200, 
                        exp_reference='nrt_eucl_run\\twoweeks_d%s\\%s_nrt_eucl_t%s_d%s+ending%s' % \
                        (d_eps, date_str, t_eps, d_eps, maxTime.replace(' ','_').replace(':','-')))