# websites domain
import os
# URL_MAPPINGS dictionary to be populated at runtime via reload_config
URL_MAPPINGS = {}

def reload_config(config_data: dict) -> None:
    """
    Populate URL_MAPPINGS from the provided configuration data.
    This replaces the eager loading that was previously at module level.
    """
    global URL_MAPPINGS
    URL_MAPPINGS.clear()

    sites_config = config_data.get("sites", {})
    if not sites_config:
        return

    active_sites = sites_config.get("active_sites", [])
    sites_with_config = sites_config.get("sites_with_config", {})

    for site_name in active_sites:
        site_cfg = sites_with_config.get(site_name)
        if site_cfg:
            url = site_cfg.get("url")
            if url:
                # Map URL to itself as requested
                URL_MAPPINGS[url] = url
    