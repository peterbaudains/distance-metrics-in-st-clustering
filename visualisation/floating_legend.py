from vizent.vizent_plot import add_point
from vizent.scales import get_shape

class FloatingLegend:

    def __init__(self, parent_ax, x0, y0, width, height, facecolor, title, fontsize=12):

        self.parent_ax = parent_ax
        self.ax = parent_ax.inset_axes([x0, y0, width, height])
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_facecolor(facecolor)
        self.ax.text(x=0.5, y=0.95, s=title, ha='center', fontsize=fontsize+2, color='w')
        self.ax.axis([0,1,0,1])
        self.fontsize=fontsize

    def add_color_elements(self, title, colorscale, colormap, x, y, s):
        self.ax.text(x=0.25, y=0.85, s=title, ha='center', fontsize=self.fontsize, color='w')
        self.ax.scatter(x=x, y=y, s=s, c=colorscale, cmap=colormap)
        colorscale_interval_length = (colorscale[-1] - colorscale[0]) / len(colorscale)
    
        self.ax.text(x=0.3, y=y[0], s="< %d" % (1+int((colorscale[1] - colorscale_interval_length / 2) / 60.0) - 1), 
                     ha='center', va='center', size=self.fontsize, color='w')
        for i in range(1,len(colorscale)-1):
            self.ax.text(x=0.3, y=y[i], s="%d - %d" % \
                         (int((colorscale[i] - colorscale_interval_length / 2) / 60.0), 1 + int((colorscale[i] + colorscale_interval_length / 2) / 60.0) - 1), 
                         ha='center', va='center', size=self.fontsize, color='w')
        self.ax.text(x=0.3, y=y[-1], s=r"> %d" % int((colorscale[-1] - colorscale_interval_length) / 60.0), 
                     ha='center', va='center', size=self.fontsize, color='w')

    def add_shape_elements(self, title, shapescale, frequencyscale, x, y, s):
        self.ax.text(x=0.75, y=0.85, s=title, ha='center', fontsize=self.fontsize, color='w')
        for i in range(len(frequencyscale) - 1):
            add_point(x[i], y[i],
                      get_shape(shapescale[i], 'sine', False, 'sine', 'sine'),
                      frequencyscale[i], '0.5', s, self.ax)
            self.ax.text(x=0.75, y=y[i], s="{:.{prec}f} - {:.{prec}f}".format(shapescale[i], shapescale[i+1], prec=1), 
                         ha='left', va='center', size=self.fontsize, color='w')