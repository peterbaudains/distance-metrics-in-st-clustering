
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))

from clustering.euclidean_dbscan import euclideanDBSCAN
from data_loader.neo4j_data_loader import DataLoaderNeo4j
from experiments.experiment import run_experiment

import datetime as dt
import logging
import time
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
    t_eps = 300
    d_eps = 25
    min_samples = 10
    
    t1 = time.time()
    df = DataLoaderNeo4j().load_df(extent=extent, minTime=minTime, maxTime=maxTime)
    t2 = time.time()
    log.info(f'Data loaded. Time taken: {t2 - t1}')

    cluster_algo = euclideanDBSCAN(d_eps=d_eps, t_eps=t_eps, min_samples=min_samples)
    run_experiment(df, cluster_algo, max_speed=maxSpeed, frame_size=10800, 
                   exp_reference='%s_euclidean_1Nov23to15Nov23_d%s_t%s' % (date_str, d_eps, t_eps), 
                   save_obs=False)

    log.info(f'Experiment complete. Total time taken: {time.time() - t1}')