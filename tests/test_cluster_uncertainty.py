import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))

import pandas as pd
from clustering.cluster_uncertainty import get_neighbourhood_data_by_cluster, calculate_cluster_certainty

def test_get_neighbourhood_data_by_cluster(cluster_df, distance_buffer):
    df = get_neighbourhood_data_by_cluster(cluster_df, distance_buffer)
    print(df.head())


def test_calculate_cluster_certainty(cluster_df):
    df = calculate_cluster_certainty(cluster_df, 
                                distance_buffer=200, 
                                speed_threshold=0.3, 
                                simplify=True)
    print(df.head())


def load_cluster_df_from_file(file):
    cluster_df = pd.read_csv(file, header=[0,1])
    cluster_df.columns = ['recordedAtTimeMin', 'recordedAtTimeMax', 'nObs', 'longitude', 'latitude', 'nVehicleUnique']
    cluster_df['recordedAtTimeMin'] = pd.to_datetime(cluster_df['recordedAtTimeMin'])
    cluster_df['recordedAtTimeMax'] = pd.to_datetime(cluster_df['recordedAtTimeMax'])

    cluster_df['recordedAtTimeMin'] = pd.to_datetime(cluster_df['recordedAtTimeMin']).dt.tz_localize(None)
    cluster_df['unix_time_min'] = ((cluster_df[['recordedAtTimeMin']] - pd.Timestamp("1970-01-01")) // \
                        pd.Timedelta('1s'))['recordedAtTimeMin'].values

    cluster_df['recordedAtTimeMax'] = pd.to_datetime(cluster_df['recordedAtTimeMax']).dt.tz_localize(None)
    cluster_df['unix_time_max'] = ((cluster_df[['recordedAtTimeMax']] - pd.Timestamp("1970-01-01")) // \
                        pd.Timedelta('1s'))['recordedAtTimeMax'].values
    return cluster_df


if __name__=="__main__": 
    cluster_df = load_cluster_df_from_file("outputs\\test_network_distance.csv")
    
    test_get_neighbourhood_data_by_cluster(cluster_df, distance_buffer=200)
    test_calculate_cluster_certainty(cluster_df)
