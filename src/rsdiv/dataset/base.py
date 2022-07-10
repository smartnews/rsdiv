import os
import zipfile
from abc import ABCMeta
from pathlib import Path
from typing import Optional, Union
from urllib.request import urlretrieve


class BaseDownloader(metaclass=ABCMeta):
    DOWNLOAD_URL: str
    DEFAULT_PATH: str

    def __init__(self, zip_path: Optional[Union[Path, str]] = None):
        if zip_path is None:
            zip_path = self.DEFAULT_PATH
        else:
            zip_path = zip_path
        self.zip_path = Path(zip_path)
        if not self.zip_path.exists():
            self._retrieve()

    def _retrieve(self) -> None:
        url: str = self.DOWNLOAD_URL
        file_name: str = str(self.zip_path) + ".zip"
        urlretrieve(url, filename=file_name)
        with zipfile.ZipFile(file_name) as zf:
            zf.extractall(self.zip_path.parent)
        os.remove(file_name)
