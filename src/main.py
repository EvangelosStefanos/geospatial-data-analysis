################################################################################
####  IMPORTS
################################################################################

import constants
# import sentinelhub
# from oauthlib.oauth2 import BackendApplicationClient
# from requests_oauthlib import OAuth2Session
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import detection

from torch.utils.data import DataLoader
from torchgeo.datasets import EnviroAtlas, stack_samples
from torchgeo.samplers import RandomGeoSampler
import torch

################################################################################
####  FILE STRUCTRURE
################################################################################

Path(constants.DATA_DIR).mkdir(exist_ok=True)
Path(constants.OUTPUT_DIR).mkdir(exist_ok=True)


def save_image(image, fname):
    import matplotlib

    dpi = matplotlib.rcParams["figure.dpi"]
    height, width, depth = image.shape

    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis("off")

    # Display the image.
    ax.imshow(image)

    plt.savefig(fname=fname)
    plt.close(fig)
    return


torch.manual_seed(0)
dataset = EnviroAtlas(root="/app/data", download=True)
sampler = RandomGeoSampler(dataset, size=1024, length=10)
dataloader = DataLoader(dataset, sampler=sampler, collate_fn=stack_samples)

for i, sample in enumerate(dataloader):
    image = sample['image']
    image = np.ascontiguousarray(image[0,:3,:,:].T, dtype=np.uint8) # unnormalized BGRA (b, c, h, w) -> contiguous unnormalized RGB (w, h, c)

    results = detection.detect(image)

    fname = f"{constants.OUTPUT_DIR}/{i}.jpg"
    save_image(image=results["annotated_image"], fname=fname)
    

exit(0)


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

# aoi_coords_wgs84 = [22.962284, 40.525283, 23.087597, 40.567220]  # thermi bbox
# aoi_coords_wgs84 = [22.902288,40.626754,22.939453,40.642940]  # thess port
aoi_coords_wgs84 = [22.929239,40.568328,22.960224,40.592969]  # kalamaria
# aoi_coords_wgs84 = [23.00, 40.54, 23.04, 40.58]  # thermi bbox
resolution = 10
aoi_bbox = sentinelhub.BBox(bbox=aoi_coords_wgs84, crs=sentinelhub.CRS.WGS84)
aoi_size = sentinelhub.bbox_to_dimensions(aoi_bbox, resolution=resolution)

print(f"Image shape at {resolution} m resolution: {aoi_size} pixels")


################################################################################
####  PROCESSING API  https://documentation.dataspace.copernicus.eu/notebook-samples/sentinelhub/introduction_to_SH_APIs.html#example-1-true-color-image
################################################################################
def save_image(image, fname):
    import matplotlib

    dpi = matplotlib.rcParams["figure.dpi"]
    height, width, depth = image.shape

    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis("off")

    # Display the image.
    ax.imshow(image)

    plt.savefig(fname=fname)
    plt.close(fig)
    return


import processing
import evalscripts

start_date = {
    "year" : 2023,
    "month" : 1,
}
end_date = {
    "year": 2023,
    "month": 6,
}

START_DATE_DAY = 1
END_DATE_DAY = 25
time_intervals = []
for year in range(start_date["year"], end_date["year"]+1):
    start_month = 1
    end_month = 12
    if year == start_date["year"]:
        start_month = start_date["month"]
    if year == end_date["year"]:
        end_month = end_date["month"]
    month_range = range(start_month, end_month+1)
    for month in month_range:
        time_intervals += [(
            f"{year}-{month:02d}-{START_DATE_DAY:02d}", 
            f"{year}-{month:02d}-{END_DATE_DAY:02d}"
            )]
for i, time_interval in enumerate(time_intervals):
    ####  true color image

    response = processing.make_request(
        evalscript=evalscripts.TRUE_COLOR,
        time_interval=time_interval,
        aoi_bbox=aoi_bbox,
        aoi_size=aoi_size,
        config=config,
    )

    image = response[0]
    save_image(
        np.clip(image * 3.5 / 255, 0, 1),
        f"{constants.OUTPUT_DIR}/thermi_true_color_{i}.png",
    )
    print(time_interval)
    # image = get_image(time_interval, f"{constants.OUTPUT_DIR}/detection_{i}.png")
    image = np.clip((3.5 / 255.0) * image, 0, 1)
    image = detection.detect(image)
    save_image(image=image, fname=f"{constants.OUTPUT_DIR}/detection_{i}.jpeg")
    # detect_objects(image)
    # detect_changes(image, image_previous)

exit(0)

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
