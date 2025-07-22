import hashlib
import os
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import tqdm
from segmentation_robustness_framework.utils.dataset_utils import (
    _file_md5,
    _progress_hook,
    download,
    extract,
    get_cache_dir,
)


def test_progress_hook():
    mock_tqdm = Mock(spec=tqdm.tqdm)
    hook = _progress_hook(mock_tqdm)

    hook(blocks=1, block_size=1024, total_size=2048)
    assert mock_tqdm.total == 2048
    mock_tqdm.update.assert_called_with(1024)

    hook(blocks=2, block_size=512)
    mock_tqdm.update.assert_called_with(512)


def test_progress_hook_multiple_calls():
    mock_tqdm = Mock(spec=tqdm.tqdm)
    hook = _progress_hook(mock_tqdm)

    hook(blocks=1, block_size=100, total_size=1000)
    hook(blocks=3, block_size=100)
    hook(blocks=5, block_size=100)

    expected_calls = [
        ((100,),),
        ((200,),),
        ((200,),),
    ]
    assert mock_tqdm.update.call_args_list == expected_calls


@patch("urllib.request.urlretrieve")
@patch("os.path.exists")
@patch("os.makedirs")
def test_download_new_file(mock_makedirs, mock_exists, mock_urlretrieve):
    mock_exists.return_value = False
    mock_urlretrieve.return_value = None

    with patch("tqdm.tqdm") as mock_tqdm:
        mock_tqdm_instance = Mock()
        mock_tqdm.return_value.__enter__.return_value = mock_tqdm_instance

        result = download("http://example.com/file.zip", "/tmp/test")

        mock_makedirs.assert_called_with("/tmp/test", exist_ok=True)
        mock_urlretrieve.assert_called_once()
        assert result == "/tmp/test/file.zip"


@patch("urllib.request.urlretrieve")
@patch("os.path.exists")
@patch("os.makedirs")
def test_download_existing_file_no_md5(mock_makedirs, mock_exists, mock_urlretrieve):
    mock_exists.return_value = True
    mock_urlretrieve.return_value = None

    result = download("http://example.com/file.zip", "/tmp/test")

    mock_urlretrieve.assert_not_called()
    assert result == "/tmp/test/file.zip"


@patch("urllib.request.urlretrieve")
@patch("os.path.exists")
@patch("os.makedirs")
def test_download_existing_file_with_md5_match(mock_makedirs, mock_exists, mock_urlretrieve):
    mock_exists.return_value = True
    mock_urlretrieve.return_value = None

    with patch("segmentation_robustness_framework.utils.dataset_utils._file_md5", return_value="abc123"):
        result = download("http://example.com/file.zip", "/tmp/test", md5="abc123")

    mock_urlretrieve.assert_not_called()
    assert result == "/tmp/test/file.zip"


@patch("urllib.request.urlretrieve")
@patch("os.path.exists")
@patch("os.makedirs")
def test_download_existing_file_with_md5_mismatch(mock_makedirs, mock_exists, mock_urlretrieve):
    mock_exists.return_value = True
    mock_urlretrieve.return_value = None

    with patch("tqdm.tqdm") as mock_tqdm:
        mock_tqdm_instance = Mock()
        mock_tqdm.return_value.__enter__.return_value = mock_tqdm_instance

        with patch("segmentation_robustness_framework.utils.dataset_utils._file_md5", return_value="def456"):
            result = download("http://example.com/file.zip", "/tmp/test", md5="abc123")

    mock_urlretrieve.assert_called_once()
    assert result == "/tmp/test/file.zip"


@patch("urllib.request.urlretrieve")
@patch("os.path.exists")
@patch("os.makedirs")
def test_download_kaggle_url(mock_makedirs, mock_exists, mock_urlretrieve):
    mock_exists.return_value = False
    mock_urlretrieve.return_value = None

    with patch("tqdm.tqdm") as mock_tqdm:
        mock_tqdm_instance = Mock()
        mock_tqdm.return_value.__enter__.return_value = mock_tqdm_instance

        result = download("https://kaggle.com/api/v1/datasets/download/test", "/tmp/test")

    assert result == "/tmp/test/test.zip"


def test_extract_zip_file():
    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = os.path.join(temp_dir, "test.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("test.txt", "test content")

        extract_dir = os.path.join(temp_dir, "extracted")
        extract(zip_path, extract_dir)

        extracted_file = os.path.join(extract_dir, "test.txt")
        assert os.path.exists(extracted_file)
        with open(extracted_file) as f:
            assert f.read() == "test content"


def test_extract_tar_file():
    import tarfile
    from io import BytesIO

    with tempfile.TemporaryDirectory() as temp_dir:
        tar_path = os.path.join(temp_dir, "test.tar.gz")
        with tarfile.open(tar_path, "w:gz") as tf:
            test_content = b"test content"
            tarinfo = tarfile.TarInfo("test.txt")
            tarinfo.size = len(test_content)
            fileobj = BytesIO(test_content)
            tf.addfile(tarinfo, fileobj=fileobj)

        extract_dir = os.path.join(temp_dir, "extracted")
        extract(tar_path, extract_dir)

        assert os.path.exists(extract_dir)


def test_extract_unsupported_format():
    with tempfile.TemporaryDirectory() as temp_dir:
        unsupported_file = os.path.join(temp_dir, "test.xyz")
        with open(unsupported_file, "w") as f:
            f.write("test")

        extract_dir = os.path.join(temp_dir, "extracted")

        extract(unsupported_file, extract_dir)
        assert os.path.exists(extract_dir)


def test_file_md5():
    with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
        f.write(b"test content")
        temp_file = f.name

    try:
        md5_hash = _file_md5(temp_file)
        expected_hash = hashlib.md5(b"test content").hexdigest()
        assert md5_hash == expected_hash
    finally:
        os.unlink(temp_file)


def test_file_md5_large_file():
    with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
        large_content = b"x" * (3 * (2 << 20))
        f.write(large_content)
        temp_file = f.name

    try:
        md5_hash = _file_md5(temp_file)
        expected_hash = hashlib.md5(large_content).hexdigest()
        assert md5_hash == expected_hash
    finally:
        os.unlink(temp_file)


def test_file_md5_custom_chunk_size():
    with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
        f.write(b"test content")
        temp_file = f.name

    try:
        md5_hash = _file_md5(temp_file, chunk=1024)
        expected_hash = hashlib.md5(b"test content").hexdigest()
        assert md5_hash == expected_hash
    finally:
        os.unlink(temp_file)


def test_get_cache_dir_default():
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch.dict(os.environ, {}, clear=True):
            with patch("pathlib.Path.home", return_value=Path(temp_dir)):
                cache_dir = get_cache_dir("TestDataset")

                expected_path = Path(temp_dir) / ".cache" / "segmentation_robustness_framework" / "testdataset"
                assert cache_dir == expected_path
                assert cache_dir.exists()


def test_get_cache_dir_xdg():
    with tempfile.TemporaryDirectory() as temp_dir:
        xdg_cache = os.path.join(temp_dir, "custom_cache")
        with patch.dict(os.environ, {"XDG_CACHE_HOME": xdg_cache}, clear=True):
            cache_dir = get_cache_dir("TestDataset")

            expected_path = Path(xdg_cache) / "segmentation_robustness_framework" / "testdataset"
            assert cache_dir == expected_path
            assert cache_dir.exists()


def test_get_cache_dir_custom():
    with tempfile.TemporaryDirectory() as temp_dir:
        custom_data = os.path.join(temp_dir, "custom_data")
        with patch.dict(os.environ, {"SEG_ROB_DATA_DIR": custom_data}, clear=True):
            cache_dir = get_cache_dir("TestDataset")

            expected_path = Path(custom_data) / "segmentation_robustness_framework" / "testdataset"
            assert cache_dir == expected_path
            assert cache_dir.exists()


def test_get_cache_dir_lowercase():
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch.dict(os.environ, {}, clear=True):
            with patch("pathlib.Path.home", return_value=Path(temp_dir)):
                cache_dir = get_cache_dir("UPPERCASE_DATASET")

                assert cache_dir.name == "uppercase_dataset"


def test_get_cache_dir_special_characters():
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch.dict(os.environ, {}, clear=True):
            with patch("pathlib.Path.home", return_value=Path(temp_dir)):
                cache_dir = get_cache_dir("Dataset-With-Special_Chars")

                assert cache_dir.name == "dataset-with-special_chars"


def test_get_cache_dir_multiple_calls():
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch.dict(os.environ, {}, clear=True):
            with patch("pathlib.Path.home", return_value=Path(temp_dir)):
                cache_dir1 = get_cache_dir("TestDataset")
                cache_dir2 = get_cache_dir("TestDataset")

                assert cache_dir1 == cache_dir2
                assert cache_dir1.exists()


@patch("urllib.request.urlretrieve")
@patch("os.path.exists")
@patch("os.makedirs")
def test_download_creates_directory(mock_makedirs, mock_exists, mock_urlretrieve):
    mock_exists.return_value = False
    mock_urlretrieve.return_value = None

    with patch("tqdm.tqdm") as mock_tqdm:
        mock_tqdm_instance = Mock()
        mock_tqdm.return_value.__enter__.return_value = mock_tqdm_instance

        download("http://example.com/file.zip", "/tmp/nonexistent/dir")

        mock_makedirs.assert_called_with("/tmp/nonexistent/dir", exist_ok=True)


def test_extract_creates_directory():
    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = os.path.join(temp_dir, "test.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("test.txt", "test content")

        extract_dir = os.path.join(temp_dir, "nonexistent", "subdir")
        extract(zip_path, extract_dir)

        assert os.path.exists(extract_dir)


def test_file_md5_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        _file_md5("/nonexistent/file.txt")


def test_progress_hook_edge_cases():
    mock_tqdm = Mock(spec=tqdm.tqdm)
    hook = _progress_hook(mock_tqdm)

    hook(blocks=0, block_size=0, total_size=0)
    assert mock_tqdm.total == 0
    mock_tqdm.update.assert_called_with(0)

    hook(blocks=1, block_size=100, total_size=None)
    assert mock_tqdm.total == 0
