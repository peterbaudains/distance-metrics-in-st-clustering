import os
import cartopy.io.img_tiles as cimgt
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from vizent.background_map import get_projected_aspects

# Load dotenv
load_dotenv()

mapbox_dark = cimgt.MapboxTiles(access_token=os.environ['MAPBOX_API_TOKEN'], map_id='dark-v11')

#extent =  [-0.154126,-0.073616,51.487048, 51.525513]
#aspx, aspy = get_projected_aspects(extent, mapbox_light.crs)
#print(aspx, aspy, aspy / aspx)

def get_bbox(centre_x, centre_y, length, desired_aspect):
    left_x = centre_x - length / 2.0
    right_x = centre_x + length / 2.0
    height = length * desired_aspect
    bottom_y = centre_y - height / 2.0
    top_y = centre_y + height / 2.0
    return [left_x, right_x, bottom_y, top_y]


def test_get_bbox():
    centre_lat = 51.508616
    centre_lon = -0.116808
    centre_x, centre_y = mapbox_dark.crs.transform_point(x=centre_lon, y=centre_lat, src_crs=ccrs.PlateCarree())
    length = 5000
    desired_aspect = 0.5625
    extent = get_bbox(centre_x, centre_y, length, desired_aspect)
    assert extent[1] - extent[0] == length
    assert extent[3] - extent[2] == desired_aspect * length


if __name__=="__main__":

    fig = plt.figure(figsize=(3840 / 172, 2160 / 172), dpi=172)
    centre_lat = 51.508616
    centre_lon = -0.116808
    centre_x, centre_y = mapbox_dark.crs.transform_point(x=centre_lon, y=centre_lat, src_crs=ccrs.PlateCarree())
    length = 10000
    desired_aspect = 0.5625
    extent = get_bbox(centre_x, centre_y, length, desired_aspect)

    print(extent)
    print(extent[1] - extent[0])
    print(extent[3] - extent[2])
    #ax1 = fig.add_subplot(111, projection=mapbox_dark.crs)
    #ax1.set_position([0, 0, 1, 1])
    #ax1.set_extent(extent, crs=mapbox_dark.crs)
    #ax1.add_image(mapbox_dark, 14)
    #plt.savefig('mapbox_background_large.png', dpi=172)#, bbox_inches='tight', pad_inches=0)

