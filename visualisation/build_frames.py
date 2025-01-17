import datetime as dt
import logging
import sys
import os

import numpy as np
import pandas as pd
import geopandas as gpd
from dotenv import load_dotenv
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from vizent.scales import get_shape_scale, get_frequency_scale, get_color_scale
import cartopy.io.img_tiles as cimgt

# Imports from within package
from data_loader.neo4j_data_loader import DataLoaderNeo4j
from visualisation.trajectories import define_trajectories
from visualisation.bus_disruption_vis import BusDisruptionVis
from visualisation.floating_legend import FloatingLegend

load_dotenv(override=True)


def load_cluster_stats(filename, crs):
    '''
    Loads aggregated cluster data resulting from clustering procedure. Each 
    row corresponds to a distinct cluster. 
    '''
    stats = pd.read_csv(filename, skiprows=1)
    stats.columns = ['cluster','recordedAtTimeMin','recordedAtTimeMax','nObs',\
                     'longitude','latitude','unixTimeMin',\
                     'unixTimeMax','nearest_node','distance','sourceNodeId',\
                     'slow_moving_observations','fast_moving_observations',\
                     'cluster_certainty']
    stats['min_time'] = pd.to_datetime(stats['recordedAtTimeMin'])
    stats['max_time'] = pd.to_datetime(stats['recordedAtTimeMax'])
    stats['duration'] = (stats['max_time'] - stats['min_time']).dt.total_seconds()
    #stats = stats[stats['unique_vehicles'] > 4].reset_index(drop=False)
    stats = stats[(stats['max_time'] > '2023-11-08 08:00:00') & (stats['min_time'] < '2023-11-08 11:30:00')].reset_index()
    stats = stats.sort_values(by='cluster_certainty').reset_index(drop=True)
    # project to mapbox dark project
    stats = gpd.GeoDataFrame(stats, geometry=gpd.points_from_xy(stats['longitude'], stats['latitude']), crs=4326)
    stats.to_crs(crs, inplace=True)
    stats['centroid_x'] = stats.geometry.x
    stats['centroid_y'] = stats.geometry.y
    stats['cluster_uncertainty'] = 1 - stats['cluster_certainty']

    return stats


def load_cluster_data(filename, crs):
    '''
    Loads pre-aggregated data resulting from clustering procedure, where each 
    row corresponds to an observation
    '''
    df_clusters = pd.read_csv(filename)
    df_clusters['recordedAtTime'] = pd.to_datetime(df_clusters['recordedAtTime'])
    df_clusters = gpd.GeoDataFrame(df_clusters, geometry=gpd.points_from_xy(df_clusters['longitude'], df_clusters['latitude']), crs=4326)
    df_clusters.to_crs(crs, inplace=True)
    df_clusters['eastings'] = df_clusters.geometry.x
    df_clusters['northings'] = df_clusters.geometry.y
    df_clusters = df_clusters[df_clusters['cluster'] >= 0]
    stats = df_clusters.groupby('cluster').agg({'recordedAtTime': ["min", "max", np.size], 'eastings': ["mean"], 'northings': ["mean"], 'vehicleRef': ['nunique']})
    stats.columns = ['min_time', 'max_time', 'observations', 'centroid_x', 'centroid_y', 'unique_vehicles']
    stats['duration'] = (stats['max_time'] - stats['min_time']).dt.total_seconds()
    stats = stats[stats['unique_vehicles'] > 4].reset_index(drop=False)
    stats = stats.sort_values(by='unique_vehicles').reset_index(drop=True)
    return stats


def draw_legend(bdp, stats, colormap):
    imax = bdp.set_background("visualisation/images/mapbox_background_large.png")
    imax.remove()
    bdp.ax.patch.set_alpha(0)
    bdp.fig.patch.set_alpha(0)
    
    # Get scales for legend
    colorscale = get_color_scale(stats['duration'], 3600, 0, 3, None)
    shapescale = get_shape_scale(np.ravel(stats['cluster_certainty']), 1, 0, 6, False, None)

    frequencyscale = get_frequency_scale(shapescale, False)

    legend = FloatingLegend(parent_ax=bdp.ax, 
                            x0=0.02, 
                            y0=0.25, 
                            width=0.15, 
                            height=0.5, 
                            facecolor='0.1', 
                            title="Key for Congestion Clusters")
    
    legend.add_color_elements(title="Cluster duration\n(minutes)", 
                              colorscale=colorscale,
                              colormap=colormap, 
                              y=[0.26, 0.42, 0.58], 
                              x=[0.1 for i in range(3)], 
                              s=900)

    legend.add_shape_elements(title="Cluster uncertainty", 
                              shapescale=shapescale, 
                              frequencyscale=frequencyscale, 
                              x=[0.6 for i in range(5)], 
                              y=[0.1, 0.26, 0.42, 0.58, 0.74], 
                              s=50)
    return legend



def build_frames():

    # Define MapboxTiles instance for map projection.
    mapbox_dark = cimgt.MapboxTiles(access_token=os.environ['MAPBOX_API_TOKEN'], 
                                    map_id='dark-v11')

    bdp = BusDisruptionVis(figsize=(3840/172, 2160/172), 
                            dpi=172, 
                            projection=mapbox_dark, 
                            centre_lat = 51.508616,
                            centre_lon = -0.116808, 
                            range=10000)

    # Add background and logos
    imax = bdp.set_background("visualisation/images/mapbox_background_large.png")
    
    # Save background plot
    bdp.fig.savefig(os.environ['BUSDISVIZ_OUTDIR'] + "background.png")
    
    # Remove background and prepare logo figure
    imax.remove()
    bdp.add_attribution('visualisation/images/mapbox-logo-white.png')
    # bdp.add_logo('visualisation/images/CUSP_LOGO_HIGH_RES.png')
    # save figure with logos and attribution
    bdp.fig.savefig(os.environ['BUSDISVIZ_OUTDIR'] + "logo.png", transparent=True)
    
    # Remove logo and prepare legend
    bdp.ax.cla()
    stats = load_cluster_stats('..\\distance-metrics-in-st-clustering\\outputs\\20241211_network_test_d50_t300.csv', mapbox_dark.crs)
    colormap_colors = np.array([cm.YlOrRd(0.15), cm.YlOrRd(0.7), cm.YlOrRd(0.9)])
    colormap = ListedColormap(colormap_colors)
    legend = draw_legend(bdp, stats, colormap)    
    bdp.fig.savefig(os.environ['BUSDISVIZ_OUTDIR'] + "legend.png", 
                    facecolor=bdp.fig.get_facecolor(), 
                    edgecolor='none')
    legend.ax.remove()

    film_start = dt.datetime(2023, 11, 8, 8, 30)
    film_end = dt.datetime(2023, 11, 8, 11, 30)
    frames = 2160
    frame_range = range(frames)
    #frame_range = [1091]
    frame_fade = 3
    time_index = pd.date_range(start=pd.Timestamp(film_start), 
                               end=pd.Timestamp(film_end), 
                               periods=frames)

    bdp.add_title(time_index[0])

    frame_duration = (film_end - film_start).total_seconds() / frames
    log.info('Frame duration in data time: %.4f seconds' % frame_duration)
    log.info('Fade duration: %.4f seconds' % (frame_fade * frame_duration))
    
    # for trajectory data, we need to load the data from neo4j
    df = DataLoaderNeo4j().load_df(extent=bdp.extent_wgs, 
                                   minTime="2023-11-08T08:00:00Z", 
                                   maxTime="2023-11-08T11:30:00Z")

    # # Loop through frames
    # for frame in frame_range:
    #     bdp.update_title(frame, time_index)
    #     bdp.fig.savefig(os.environ['BUSDISVIZ_OUTDIR'] + "titles\\" + str(frame).zfill(4) + ".png", transparent=True)

    bdp.ax.cla()
    imax = bdp.set_background("visualisation/images/mapbox_background_large.png")
    bdp.add_cluster_glyphs(stats, colormap, shape_n=5, color_n=3)
    imax.remove()
    # Loop through frames
    for frame in frame_range:
        log.info("Glyphs frame: %s" % frame)
        bdp.update_glyphs(frame, time_index)
        bdp.fig.savefig(os.environ['BUSDISVIZ_OUTDIR'] + "glyphs\\" + str(frame).zfill(4) + ".png", transparent=True, bbox_inches='tight', pad_inches=0)
    
    bdp.ax.cla()
    imax = bdp.set_background("visualisation/images/mapbox_background_large.png")
    # Define trajectories
    tdf = define_trajectories(df, dt.datetime(2023, 11, 8, 8, 0), film_end, mapbox_dark)
    bdp.add_trajectories(tdf, frame_fade=3)
    imax.remove()

    # Loop through frames
    for frame in frame_range:
        log.info("Trajectories frame %s" % frame)
        bdp.update_trajectories(frame, time_index)
        bdp.fig.savefig(os.environ['BUSDISVIZ_OUTDIR'] + "trajectories\\" + str(frame).zfill(4) + ".png", transparent=True)
    

if __name__ == "__main__":

    # Configure logging
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    log = logging.getLogger()

    build_frames()