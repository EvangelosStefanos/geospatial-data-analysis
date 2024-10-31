################################################################################
####  IMPORTS
################################################################################

import constants
import sentinelhub
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session
import matplotlib.pyplot as plt
from pathlib import Path

################################################################################
####  FILE STRUCTRURE
################################################################################

Path(constants.DATA_DIR).mkdir(exist_ok=True)
Path(constants.OUTPUT_DIR).mkdir(exist_ok=True)

################################################################################
####  SET UP CONFIGURATION PROFILE  https://documentation.dataspace.copernicus.eu/notebook-samples/sentinelhub/introduction_to_SH_APIs.html#credentials
################################################################################

import configuration

config = configuration.get_config()  # create or get existing profile

################################################################################
####  REQUEST TOKEN  https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Overview/Authentication.html#python
################################################################################

# Create a session
client = BackendApplicationClient(client_id=config.sh_client_id)
oauth = OAuth2Session(client=client)

# Get token for the session
token = oauth.fetch_token(
    token_url="https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
    client_secret=config.sh_client_secret,
    include_client_id=True,
)

# All requests using this session will have an access token automatically added
resp = oauth.get("https://sh.dataspace.copernicus.eu/configuration/v1/wms/instances")
print(resp.content)


def sentinelhub_compliance_hook(response):
    response.raise_for_status()
    return response


oauth.register_compliance_hook("access_token_response", sentinelhub_compliance_hook)


################################################################################
####  SET AREA OF INTEREST  https://documentation.dataspace.copernicus.eu/notebook-samples/sentinelhub/introduction_to_SH_APIs.html#setting-an-area-of-interest
################################################################################

aoi_coords_wgs84 = [22.962284, 40.525283, 23.087597, 40.567220]  # thermi bbox
resolution = 10
aoi_bbox = sentinelhub.BBox(bbox=aoi_coords_wgs84, crs=sentinelhub.CRS.WGS84)
aoi_size = sentinelhub.bbox_to_dimensions(aoi_bbox, resolution=resolution)

print(f"Image shape at {resolution} m resolution: {aoi_size} pixels")

################################################################################
####  CATALOG  https://documentation.dataspace.copernicus.eu/notebook-samples/sentinelhub/introduction_to_SH_APIs.html#catalog-api
################################################################################

# catalog = sentinelhub.SentinelHubCatalog(config=config)
# aoi_bbox = sentinelhub.BBox(bbox=aoi_coords_wgs84, crs=sentinelhub.CRS.WGS84)
# time_interval = "2020-01-01", "2024-01-01"

# search_iterator = catalog.search(
#     sentinelhub.DataCollection.SENTINEL2_L2A,
#     bbox=aoi_bbox,
#     time=time_interval,
#     fields={"include": ["id", "properties.datetime"], "exclude": []},
# )

# results = list(search_iterator)
# print("Total number of results:", len(results))
# print(results)

################################################################################
####  PROCESSING API  https://documentation.dataspace.copernicus.eu/notebook-samples/sentinelhub/introduction_to_SH_APIs.html#example-1-true-color-image
################################################################################


import processing
import evalscripts


time_interval = ("2020-01-01", "2024-01-01")

####  true color image

response = processing.make_request(
    evalscript=evalscripts.TRUE_COLOR,
    time_interval=time_interval,
    aoi_bbox=aoi_bbox,
    aoi_size=aoi_size,
    config=config,
)
image = response[0]
processing.plot_image(image, factor=3.5 / 255, clip_range=(0, 1))
plt.savefig(fname=constants.OUTPUT_DIR + "/thermi_true_color.png")

####  ndbi image

response = processing.make_request(
    evalscript=evalscripts.NDBI_COLOR,
    time_interval=time_interval,
    aoi_bbox=aoi_bbox,
    aoi_size=aoi_size,
    config=config,
)
image = response[0]
processing.plot_image(image, factor=1.0 / 255)
plt.savefig(fname=constants.OUTPUT_DIR + "/thermi_ndbi_color.png")

plt.figure()
plt.imshow(image * 1.0 / 255, cmap="terrain")
plt.savefig(fname=constants.OUTPUT_DIR + "/thermi_ndbi_color_terrain.png")


################################################################################
####  STATISTICAL API - NDBI TIME SERIES  https://documentation.dataspace.copernicus.eu/notebook-samples/sentinelhub/introduction_to_SH_APIs.html#statistical-api
################################################################################

import statistical

geometry = {
    "coordinates": [
        [
            [23.00169258340594, 40.538881638996],
            [23.037717284051723, 40.538881638996],
            [23.037717284051723, 40.555981411974955],
            [23.00169258340594, 40.555981411974955],
            [23.00169258340594, 40.538881638996],
        ]
    ],
    "type": "Polygon",
}

geometry = sentinelhub.Geometry(geometry=geometry, crs=sentinelhub.CRS.WGS84)

time_interval = ("2020-01-01T00:00:00Z", "2024-01-01T00:00:00Z")
aggregation_interval = "P365D"
size = [800, 600]

response = statistical.make_request(
    evalscript=evalscripts.NDBI,
    geometry=geometry,
    config=config,
    time_interval=time_interval,
    aggregation_interval=aggregation_interval,
    size=size,
)
result = statistical.read_acquisitions_stats(response[0]["data"])
print(result)
statistical.plot_and_save(result, constants.OUTPUT_DIR + "/thermi_ndbi_time_series.png")
