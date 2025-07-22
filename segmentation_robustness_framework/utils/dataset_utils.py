import hashlib
import os
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable

import tqdm


def _progress_hook(t: tqdm.tqdm) -> Callable:
    last_b = [0]

    def inner(blocks: int = 1, block_size: int = 1, total_size: int | None = None) -> None:
        if total_size is not None:
            t.total = total_size
        t.update((blocks - last_b[0]) * block_size)
        last_b[0] = blocks

    return inner


def download(url: str, dest_dir: str, md5: str | None = None) -> str:
    """Download URL to `dest_dir` (if not present).

    Args:
        url: URL to download
        dest_dir: Directory to save the file
        md5: Optional MD5 hash to verify download

    Returns:
        str: Full path to the downloaded file
    """
    os.makedirs(dest_dir, exist_ok=True)

    filename = os.path.basename(url)
    if "kaggle.com/api/v1" in url:
        filename += ".zip"
    fpath = os.path.join(dest_dir, filename)

    if not os.path.exists(fpath) or (md5 and _file_md5(fpath) != md5):
        with tqdm.tqdm(unit="B", unit_scale=True, desc=f"Downloading {filename}") as t:
            urllib.request.urlretrieve(url, fpath, _progress_hook(t))

    return fpath


def extract(file_path: str, dest_dir: str) -> None:
    """Extract archive to `dest_dir`.

    Args:
        file_path: Full path to the archive file
        dest_dir: Directory to extract to
    """
    os.makedirs(dest_dir, exist_ok=True)

    if file_path.endswith(".zip"):
        with zipfile.ZipFile(file_path) as zf:
            zf.extractall(dest_dir)
    elif file_path.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(file_path) as tf:
            tf.extractall(dest_dir, filter="data")


def _file_md5(path: str, chunk: int = 2 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def get_cache_dir(dataset_name: str) -> Path:
    """Return an OS-appropriate cache directory for *dataset_name*.

    The root directory is resolved as follows (first match wins):

    1. Environment variable ``SEG_ROB_DATA_DIR`` – allows users to override the
       location (e.g. to put data on an external disk).
    2. ``${XDG_CACHE_HOME}``/segmentation_robustness_framework if the XDG spec
       variable is set (Linux/Mac).
    3. ``~/.cache/segmentation_robustness_framework`` as a sensible default.

    A sub-directory named after the *lower-cased* ``dataset_name`` is appended.
    The resulting directory is created (``parents=True, exist_ok=True``) and
    returned as a ``pathlib.Path`` object.
    """

    base = os.getenv("SEG_ROB_DATA_DIR")
    if base is None:
        base = os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")
    cache_root = Path(base) / "segmentation_robustness_framework"
    cache_root.mkdir(parents=True, exist_ok=True)

    ds_dir = cache_root / dataset_name.lower()
    ds_dir.mkdir(parents=True, exist_ok=True)
    return ds_dir
