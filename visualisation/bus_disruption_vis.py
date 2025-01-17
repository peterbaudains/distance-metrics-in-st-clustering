import logging
import cartopy.crs as ccrs
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.figure import Figure
from matplotlib.image import imread
from matplotlib.pylab import f
from visualisation.map_image import get_bbox
from vizent import add_glyphs

log = logging.getLogger()


class BusDisruptionVis:
    
    def __init__(self, figsize, dpi, projection, centre_lat, centre_lon, range):
        self.fig = Figure(figsize=figsize, dpi=dpi)
        self.projection = projection
        gs = GridSpec(1, 3, width_ratios=[3,0,0])
        self.ax = self.fig.add_subplot(gs[0], projection=self.projection.crs)
        self.ax2 = self.fig.add_subplot(gs[1])
        self.ax3 = self.fig.add_subplot(gs[2])
        
        # Stretch the subplot to full screen
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0.0, wspace=0, hspace=0)
        
        # For the bus disruption viz, we are building our own bespoke legends
        # so we delete the two axes generated automatically
        self.fig.delaxes(self.ax2)
        self.fig.delaxes(self.ax3)
        self.asp = 0.5625
        self.vizent_tuple = (self.fig, self.ax, self.ax2, self.ax3, self.asp)

        centre_x, centre_y = self.projection.crs.transform_point(x=centre_lon, 
                                                                 y=centre_lat, 
                                                                 src_crs=ccrs.PlateCarree())
        self.extent_projected = get_bbox(centre_x, 
                                         centre_y, 
                                         range, 
                                         self.asp)
        
        extent_pnt1 = ccrs.PlateCarree().transform_point(x=self.extent_projected[0], y=self.extent_projected[2], src_crs=self.projection.crs)
        extent_pnt2 = ccrs.PlateCarree().transform_point(x=self.extent_projected[1], y=self.extent_projected[3], src_crs=self.projection.crs)
        self.extent_wgs = [extent_pnt1[0], extent_pnt2[0], extent_pnt1[1], extent_pnt2[1]]

    def set_background(self, image_file):
        img = imread(image_file)
        imax = self.ax.imshow(img, extent=self.extent_projected)
        return imax

    def add_attribution(self, image_file):
        im = imread(image_file)
        imax = self.ax.inset_axes((-0.015, 0.02, 0.15, 0.035))
        imax.imshow(im)
        imax.axis('off')
        # Copyright statement
        self.ax.text(x=0.87, y=0.03, s="© Mapbox, © OpenStreetMap", 
                    horizontalalignment='left',
                    verticalalignment='top',
                    fontsize=12,
                    transform=self.ax.transAxes,
                    color='w')       

    def add_logo(self, image_file):
        im = imread(image_file)
        logoax = self.ax.inset_axes((0.0, 0.86, 0.1, 0.1))
        logoax.imshow(im)
        logoax.axis('off')

    def add_cluster_glyphs(self, cluster_stats, colormap, shape_n, color_n):
        self.cluster_stats = cluster_stats
        
        # Call add_glyphs from vizent - plot all clusters to begin.
        self.glyphs = add_glyphs(self.vizent_tuple, 
                        x_values=self.cluster_stats['centroid_x'], 
                        y_values=self.cluster_stats['centroid_y'], 
                        color_values=self.cluster_stats['duration'],
                        shape_values=self.cluster_stats['cluster_uncertainty'],
                        size_values=[40 for i in range(self.cluster_stats.shape[0])],
                        colormap=colormap,
                        scale_diverges=False, 
                        color_min=0, 
                        color_max=3600, 
                        shape_min=0,
                        shape_max=1,
                        shape_n=shape_n,
                        color_n=color_n, 
                        interval_type='limit')
        #v1 size: 30

        # Set alpha to zero
        for glyph in self.glyphs:
            glyph['outer'].set_alpha(0)
            glyph['shape'].set_alpha(0)
            glyph['inner'].set_alpha(0)

    def add_title(self, text):
        self.title = self.ax.text(x=0.5, 
                                y=0.7, 
                                s=text, 
                                horizontalalignment='center', 
                                transform=self.ax.transAxes, 
                                color='w', 
                                fontsize=12)

    def add_trajectories(self, trajectories, frame_fade):
        log.info('Adding trajectories')
        self.trajectories = trajectories
        self.trajectory_paths = []
        self.frame_fade = frame_fade
        for lag in range(frame_fade):
            start_locations = trajectories.get_start_locations().geometry
            self.trajectory_paths.append(
                self.ax.scatter(start_locations.apply(lambda p: p.x), 
                                start_locations.apply(lambda p: p.y), 
                                s=10, 
                                alpha=0, 
                                linewidth=0, 
                                color='orange')
            )
        log.info('Trajectories added')


    def update_trajectories(self, frame, time_index):
        if self.trajectory_paths is None:
            log.info('No trajectories to update')
            return 
        # Update trajectories with lagged traces
        for lag in range(self.frame_fade):
            x = []
            y = []
            # for each frame, update the data stored on each artist.
            if frame - lag >= 0:
                lagged_frametime = time_index[frame - lag]
                locations = self.trajectories.get_locations_at(lagged_frametime, method='nearest')
                if locations.shape[0] > 0:
                    for traj_id in locations['traj_id'].values:
                        traj = self.trajectories.get_trajectory(traj_id)
                        pos = traj.get_position_at(lagged_frametime)
                        x.append(pos.x)
                        y.append(pos.y)
                # update the scatter plot:
            data = np.stack([x, y]).T
            self.trajectory_paths[lag].set_offsets(data)
            self.trajectory_paths[0].set_alpha(1)
            self.trajectory_paths[1].set_alpha(0.4)
            self.trajectory_paths[2].set_alpha(0.2)

                
    def update_glyphs(self, frame, time_index):
        if self.glyphs is None:
            log.info('No glyphs to update')
            return

        frametime = time_index[frame]

        # Set alphas for the glyphs if they are in the current set of clusters
        previous_clusters = self.cluster_stats[(self.cluster_stats['min_time']<time_index[frame - 1]) & \
                                               (self.cluster_stats['max_time']>time_index[frame-1])]
        for cluster_id in previous_clusters.index.tolist():
            self.glyphs[cluster_id]['outer'].set_alpha(0)
            self.glyphs[cluster_id]['shape'].set_alpha(0)
            self.glyphs[cluster_id]['inner'].set_alpha(0)

        current_clusters = self.cluster_stats[(self.cluster_stats['min_time']<frametime) & \
                                              (self.cluster_stats['max_time']>frametime)]
        for cluster_id in current_clusters.index.tolist():
            self.glyphs[cluster_id]['outer'].set_alpha(1.0)
            self.glyphs[cluster_id]['shape'].set_alpha(1.0)
            self.glyphs[cluster_id]['inner'].set_alpha(1.0)


    def update_title(self, frame, time_index):
        self.title.set_text(time_index[frame].strftime('%Y-%m-%d %H:%M:%S'))

    def update_plot(self, frame, time_index):
        log.info('Updating plot, frame: %s' % frame)
        self.update_trajectories(frame, time_index)
        self.update_glyphs(frame, time_index)
        self.update_title(frame, time_index)

