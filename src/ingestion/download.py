import os
import requests

from config import SOURCE_URL


def download_source(output_file: str, url: str = SOURCE_URL) -> str:
    """
    Download the Romanian fiscal code HTML from ANAF if it isn't already present locally.

    @param output_file Path to save the HTML to (e.g. input/Legea nr.227_2015.html)
    @param url Source URL to fetch
    @return output_file path
    """
    if os.path.exists(output_file):
        return output_file

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    response = requests.get(url, timeout=60)
    response.raise_for_status()

    with open(output_file, "wb") as f:
        f.write(response.content)

    return output_file
