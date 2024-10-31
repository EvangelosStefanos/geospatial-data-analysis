import sentinelhub


def get_config():
    config = sentinelhub.SHConfig()
    config.sh_client_id = client_id # change to your client id
    config.sh_client_secret = client_secret # change to your client secret
    config.sh_token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    config.sh_base_url = "https://sh.dataspace.copernicus.eu"
    # config.save("geospatial-data-analysis-profile")
    return config
