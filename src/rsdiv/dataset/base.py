from abc import ABCMeta
from pathlib import Path
from typing import Optional, Union
from urllib.request import urlretrieve
import zipfile

class BaseDownloader(metaclass=ABCMeta):
    """Base downloader for all Movielens datasets."""

    DOWNLOAD_URL: str
    DEFAULT_PATH: str

    def __init__(self, zip_path: Optional[Union[Path, str]] = None):
        self.zip_path = Path(zip_path or self.DEFAULT_PATH)
        if not self.zip_path.exists():
            self._retrieve()

    def _retrieve(self) -> None:
        if self.zip_path.exists():
            return

        zip_file_name = self.zip_path.with_suffix('.zip')
        urlretrieve(self.DOWNLOAD_URL, filename=zip_file_name)

        with zipfile.ZipFile(zip_file_name) as zf:
            zf.extractall(self.zip_path.parent)

        zip_file_name.unlink()
