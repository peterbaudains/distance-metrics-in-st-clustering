import logging
import time
import geopandas as gpd
import movingpandas as mpd
import cartopy.crs as ccrs

log = logging.getLogger()

def define_trajectories(df, start_time, end_time, projection):

    log.info('Defining trajectories')
    
    t = time.time()

    transformed = projection.crs.transform_points(src_crs=ccrs.PlateCarree(), 
                                                        x=df['lon2'].values, 
                                                        y=df['lat2'].values)
    df['t_x'] = [i[0] for i in transformed]
    df['t_y'] = [i[1] for i in transformed]

    # Restrict to period of interest
    df = df[(df['recordedAtTime'] > start_time) & \
            (df['recordedAtTime'] <= end_time)]
        
    # Convert to collection of trajectories
    df['traj_id'] = df.groupby(['vehicleRef', 'directionRef', 'vehicleJourneyRef', 'lineRef']).ngroup().copy()
    df = gpd.GeoDataFrame(df, 
                          geometry=gpd.points_from_xy(df['t_x'], df['t_y']), 
                          crs=projection.crs)
    
    df = df.set_index('recordedAtTime')
    tdf = mpd.TrajectoryCollection(df, 'traj_id', crs=df.crs, x='t_x', y='t_y')
    log.info('Trajectories identified -- time taken: %s' % (time.time() - t))
    return tdf


if __name__=="__main__":
    import datetime as dt
    import os
    import cartopy.io.img_tiles as cimgt
    from neo4j_data_loader import DataLoaderNeo4j

    start_time = dt.datetime(2023, 11, 8, 8, 30)
    end_time = dt.datetime(2023, 11, 8, 11, 30)
    mapbox_dark = cimgt.MapboxTiles(access_token=os.environ['MAPBOX_API_TOKEN'], 
                                    map_id='dark-v11')

    df = DataLoaderNeo4j().load_df(date=20231108, 
                                   extent=[-0.134808,-0.099263, 51.499545, 51.517922],
                                   minTime=83000, 
                                   maxTime=113000)
    
    tdf = define_trajectories(df, start_time, end_time, mapbox_dark)
    timestamped_locations = tdf.get_locations_at(dt.datetime(2023, 11, 8, 9),
                                                 method='nearest')

    count = 0
    for id in timestamped_locations['traj_id'].values:
        count += 1
        print(count, tdf.get_trajectory(id).get_position_at(dt.datetime(2023, 11, 8, 9)))
    
