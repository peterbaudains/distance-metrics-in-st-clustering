
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))

from clustering.euclidean_dbscan import euclideanDBSCAN
from data_loader.neo4j_data_loader import DataLoaderNeo4j
from experiment import run_experiment

import datetime as dt
import logging
log = logging.getLogger(__name__)

if __name__ == "__main__":

    date = dt.datetime.now()
    date_str = date.strftime("%Y%m%d")
    logging.basicConfig(filename="logs/eucl_experiment_%s.log" % date_str, 
                        filemode='a', level=logging.INFO)
    
    extent = [-0.16172376,-0.07189224,51.49288835,51.52433822]
    minTime = "2023-11-01"
    maxTime = "2023-11-15"
    maxSpeed = 0.3
    d_eps = 25
    t_eps = 300
    min_samples = 10
    df = DataLoaderNeo4j().load_df(extent=extent, minTime=minTime, maxTime=maxTime)
    
    cluster_algo = euclideanDBSCAN(d_eps=d_eps, t_eps=t_eps, min_samples=min_samples)
    run_experiment(df, cluster_algo, frame_size=10800, exp_reference='%s_eucl_test_t%s_d%s' % (date_str, t_eps, d_eps))