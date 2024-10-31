import sentinelhub
import pandas as pd
import matplotlib.pyplot as plt
import constants


################################################################################
####  STATISTICAL API
################################################################################


def make_request(
    evalscript, geometry, config, time_interval, aggregation_interval, size
):
    request = sentinelhub.SentinelHubStatistical(
        aggregation=sentinelhub.SentinelHubStatistical.aggregation(
            evalscript=evalscript,
            time_interval=time_interval,
            aggregation_interval=aggregation_interval,
            size=size,
        ),
        input_data=[
            sentinelhub.SentinelHubStatistical.input_data(
                sentinelhub.DataCollection.SENTINEL2_L1C.define_from(
                    name="s2l1c", service_url="https://sh.dataspace.copernicus.eu"
                ),
                other_args={"dataFilter": {"maxCloudCoverage": 10}},
            ),
        ],
        geometry=geometry,
        config=config,
        data_folder="data",
    )
    return request.get_data(
        save_data=True, redownload=constants.REDOWNLOAD, show_progress=True
    )


# define functions to extract statistics for all acquisition dates
def extract_stats(date, stat_data):
    d = {}
    for key, value in stat_data["outputs"].items():
        stats = value["bands"]["B0"]["stats"]
        if stats["sampleCount"] == stats["noDataCount"]:
            continue
        else:
            d["date"] = [date]
            for stat_name, stat_value in stats.items():
                if stat_name == "sampleCount" or stat_name == "noDataCount":
                    continue
                else:
                    d[f"{key}_{stat_name}"] = [stat_value]
    return pd.DataFrame(d)


def read_acquisitions_stats(stat_data):
    df_li = []
    for aq in stat_data:
        date = aq["interval"]["from"][:10]
        df_li.append(extract_stats(date, aq))
    return pd.concat(df_li)


def plot_and_save(data, fname):
    fig_stat, ax_stat = plt.subplots(1, 1, figsize=(12, 6))
    t1 = data["date"]
    index_mean = data["ndbi_mean"]
    index_std = data["ndbi_stDev"]
    ax_stat.plot(t1, index_mean, label="mean")
    ax_stat.fill_between(
        t1,
        index_mean - index_std,
        index_mean + index_std,
        alpha=0.3,
        label="std",
    )
    ax_stat.tick_params(axis="x", labelrotation=30, labelsize=12)
    ax_stat.tick_params(axis="y", labelsize=12)
    ax_stat.set_xlabel("Date", size=15)
    ax_stat.set_ylabel("NDBI/unitless", size=15)
    ax_stat.legend(loc="lower right", prop={"size": 12})
    ax_stat.set_title("NDBI time series", fontsize=20)
    for label in ax_stat.get_xticklabels()[1::2]:
        label.set_visible(False)
    plt.savefig(fname=fname)
    return


def stats_to_df(stats_data):
    """Transform Statistical API response into a pandas.DataFrame"""
    df_data = []

    for single_data in stats_data["data"]:
        df_entry = {}
        is_valid_entry = True

        df_entry["interval_from"] = sentinelhub.parse_time(
            single_data["interval"]["from"]
        ).date()
        df_entry["interval_to"] = sentinelhub.parse_time(
            single_data["interval"]["to"]
        ).date()

        for output_name, output_data in single_data["outputs"].items():
            for band_name, band_values in output_data["bands"].items():
                band_stats = band_values["stats"]
                if band_stats["sampleCount"] == band_stats["noDataCount"]:
                    is_valid_entry = False
                    break

                for stat_name, value in band_stats.items():
                    col_name = f"{output_name}_{band_name}_{stat_name}"
                    if stat_name == "percentiles":
                        for perc, perc_val in value.items():
                            perc_col_name = f"{col_name}_{perc}"
                            df_entry[perc_col_name] = perc_val
                    else:
                        df_entry[col_name] = value

        if is_valid_entry:
            df_data.append(df_entry)

    return pd.DataFrame(df_data)
