import os
import zipfile
from abc import ABCMeta
from typing import Optional
from urllib.request import urlretrieve


class BaseDownloader(metaclass=ABCMeta):
    DOWNLOAD_URL: str
    DEFAULT_PATH: str = os.getcwd()

    def __init__(self, zip_path: Optional[str] = None):
        if zip_path is None:
            self.zip_path = self.DEFAULT_PATH
        else:
            self.zip_path = zip_path
        self._retrieve()

    def _retrieve(self) -> None:
        url: str = self.DOWNLOAD_URL
        file_name: str = os.path.join(self.zip_path, url.split("/")[-1])
        urlretrieve(url, filename=file_name)
        with zipfile.ZipFile(file_name) as zf:
            zf.extractall()
        os.remove(file_name)
