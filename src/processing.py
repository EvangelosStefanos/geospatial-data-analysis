from __future__ import annotations
import sentinelhub
import matplotlib.pyplot as plt
from typing import Any
import numpy as np
import constants


################################################################################
####  PROCESSING API
################################################################################


def plot_image(
    image: np.ndarray,
    factor: float = 1.0,
    clip_range: tuple[float, float] | None = None,
    **kwargs: Any,
) -> None:
    """Utility function for plotting RGB images."""
    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
    if clip_range is not None:
        ax.imshow(np.clip(image * factor, *clip_range), **kwargs)
    else:
        ax.imshow(image * factor, **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])
    return


def make_request(evalscript, time_interval, aoi_bbox, aoi_size, config):
    request = sentinelhub.SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            sentinelhub.SentinelHubRequest.input_data(
                data_collection=sentinelhub.DataCollection.SENTINEL2_L2A.define_from(
                    name="s2l2a", service_url="https://sh.dataspace.copernicus.eu"
                ),
                time_interval=time_interval,
                other_args={"dataFilter": {"mosaickingOrder": "leastCC"}},
            )
        ],
        responses=[
            sentinelhub.SentinelHubRequest.output_response(
                "default", sentinelhub.MimeType.PNG
            )
        ],
        bbox=aoi_bbox,
        size=aoi_size,
        config=config,
        data_folder="data",
    )
    response = request.get_data(
        save_data=True, redownload=constants.REDOWNLOAD, show_progress=True
    )
    print(f"Returned data is of type = {type(response)} and length {len(response)}.")
    print(
        f"Single element in the list is of type {type(response[-1])} and has shape {response[-1].shape}"
    )

    print(f"Image type: {response[0].dtype}")
    return response
