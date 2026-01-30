"""Script to automatically login each website"""
import argparse
import glob
import os
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path

from playwright.sync_api import sync_playwright
HEADLESS = True
SLOW_MO = 0


def is_expired(
    storage_state: Path, url: str, keyword: str, url_exact: bool = True
) -> bool:
    """Test whether the cookie is expired"""
    if not storage_state.exists():
        return True

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=HEADLESS, slow_mo=SLOW_MO)
        context = browser.new_context(storage_state=storage_state)
        page = context.new_page()
        page.goto(url)
        time.sleep(1)
        d_url = page.url
        content = page.content()
        
    if keyword:
        return keyword not in content
    else:
        if url_exact:
            return d_url != url
        else:
            return url not in d_url


def login(site_list, auth_folder: str = "./.auth", sites_config: dict = None) -> None:
    """Login to one or more sites using credentials from sites_config.
    
    Args:
        site_list: List of site names to login to
        auth_folder: Path to store authentication state files
        sites_config: Dictionary with configuration for all sites {site_name: {url, username, password, ...}}
    """
    for site in site_list:
        assert site in sites_config, f"Site {site} configuration not found."
    
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=HEADLESS, slow_mo=SLOW_MO)
        context = browser.new_context()
        page = context.new_page()

        # Process each site in the list
        for site_name in site_list:
            # Get configuration for this specific site
            site_config = sites_config.get(site_name)
            web_usr = site_config.get("web_username")
            web_pwd = site_config.get("web_password")
            url = site_config.get("url")
            login_url = f"{url}{site_config.get('login_path')}"

            if "snipeit" in site_name:
                page.goto(login_url)
                page.get_by_placeholder("Username").click()
                page.get_by_placeholder("Username").fill(web_usr)
                page.get_by_placeholder("Password").click()
                page.get_by_placeholder("Password").fill(web_pwd)
                page.get_by_label("Remember Me").check()
                page.get_by_role("button", name="Login").click()
            elif "espocrm" in site_name:
                page.goto(login_url)
                try:
                    page.wait_for_load_state("networkidle", timeout=10000)
                    page.get_by_label("Username").click()
                    page.get_by_label("Username").fill(web_usr)
                    page.get_by_label("Password").click()
                    page.get_by_label("Password").fill(web_pwd)
                    page.get_by_role("button", name="Log in").click()
                except:
                    page.locator("#field-userName").click()
                    page.locator("#field-userName").fill(web_usr)
                    page.locator("#field-password").click()
                    page.locator("#field-password").fill(web_pwd)
                    page.locator("#btn-login").click()
                    time.sleep(10)
                page.wait_for_url(f"{login_url}/")
            elif "zentao" in site_name:
                page.goto(f"{login_url}m=user&f=login")
                page.locator("#account").click()
                page.locator("#account").fill(web_usr)
                page.locator("#password").click()
                page.locator("#password").fill(web_pwd)
                page.get_by_text("保持登录").click()
                page.get_by_role("button", name="登录").click()
                page.wait_for_url(f"{login_url}m=my&f=index")
            elif "openproject" in site_name:
                page.goto(f"{login_url}/login")
                page.fill("#username", web_usr)
                page.fill("#password", web_pwd)
                page.locator('input.button.-highlight.button_no-margin').nth(1).click()
                page.wait_for_url(f"{login_url}/my/page")
            elif "cmdb" in site_name:
                page.goto(f"{login_url}/user/login")
                # page.get_by_label("username").click()
                # page.get_by_label("username").fill(web_usr)
                # page.get_by_label("password").click()
                # page.get_by_label("password").fill(web_pwd)
                page.locator("#username").click()
                page.locator("#username").fill(web_usr)
                page.locator("#password").click()
                page.locator("#password").fill(web_pwd)
                page.locator(".login-button").click()
                page.wait_for_url(f"{login_url}/cmdb/instances/types/6")
            elif "itop" in site_name:
                page.goto(f"{login_url}/pages/UI.php")
                # page.get_by_label("User Name").click()
                # page.get_by_label("User Name").fill(web_usr)
                # page.get_by_label("Password").click()
                # page.get_by_label("Password").fill(web_pwd)
                page.locator("#user").click()
                page.locator("#user").fill(web_usr)
                page.locator("#pwd").click()
                page.locator("#pwd").fill(web_pwd)
                page.get_by_role("button", name="Enter iTop").click()
                # page.get_by_role("button", name="Ok").click()
                # time.sleep(10)
                page.wait_for_url(f"{login_url}/pages/UI.php")
            else:
                raise ValueError(f"Site {site_name} login not implemented.")
            # time.sleep(20)
        context.storage_state(path=f"{auth_folder}/{'.'.join(site_list)}_state.json")


def get_site_comb_from_filepath(file_path: str) -> list[str]:
    comb = os.path.basename(file_path).rsplit("_", 1)[0].split(".")
    return comb


def main(auth_folder: str = "./.auth", sites_config: dict = None) -> None:
    """Main function to login to sites.
    
    Args:
        auth_folder: Path to store authentication state files
        sites_config: Configuration for all sites {site_name: {url, username, password, ...}}
    """
    if sites_config is None:
        sites_config = {}
    
    pairs = list(combinations(SITES, 2))

    with ThreadPoolExecutor(max_workers=8) as executor:
        for pair in pairs:
            # Auth doesn't work on this pair as they share the same cookie
            if "reddit" in pair and (
                "shopping" in pair or "shopping_admin" in pair
            ):
                continue
            executor.submit(
                login, list(sorted(pair)), auth_folder=auth_folder, sites_config=sites_config
            )

        for site in SITES:
            executor.submit(login, [site], auth_folder=auth_folder, sites_config=sites_config)
    
    # parallel checking if the cookies are expired  
    futures = []
    cookie_files = list(glob.glob(f"{auth_folder}/*.json"))
    with ThreadPoolExecutor(max_workers=8) as executor:
        for c_file in cookie_files:
            comb = get_site_comb_from_filepath(c_file)
            for cur_site in comb:
                url = URLS[SITES.index(cur_site)]
                keyword = KEYWORDS[SITES.index(cur_site)]
                match = EXACT_MATCH[SITES.index(cur_site)]
                future = executor.submit(
                    is_expired, Path(c_file), url, keyword, match
                )
                futures.append(future)

    for i, future in enumerate(futures):
        assert not future.result(), f"Cookie {cookie_files[i]} expired."


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--site_list", nargs="+", default=[])
    parser.add_argument("--auth_folder", type=str, default="./.auth")
    parser.add_argument("--sites_config", type=str, default="{}", help="JSON string with all sites configuration")
    args = parser.parse_args()
    
    # Parse sites_config from JSON string
    import json
    sites_config = json.loads(args.sites_config) if args.sites_config else {}
    
    if not args.site_list:
        main(auth_folder=args.auth_folder, sites_config=sites_config)
    else:
        login(args.site_list, auth_folder=args.auth_folder, sites_config=sites_config)
