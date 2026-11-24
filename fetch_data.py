"""Fetch and verify the public source table."""
import hashlib
from pathlib import Path
from urllib.request import urlopen

SOURCE = (
    "https://raw.githubusercontent.com/doylelab/rxnpredict/"
    "57e15fdb7f7483c6bf3a601df69f6ac9e5af6965/data_table.csv"
)
SHA256 = "fe310b50897e97578078558909efc2edaa7b98ad8939b1caad740a718398ed62"
DATA = Path(__file__).resolve().parent / "data" / "data_table.csv"


def verify(payload):
    if hashlib.sha256(payload).hexdigest() != SHA256:
        raise ValueError("Source table checksum mismatch")


if __name__ == "__main__":
    with urlopen(SOURCE, timeout=30) as response:
        payload = response.read()
    verify(payload)
    DATA.parent.mkdir(exist_ok=True)
    DATA.write_bytes(payload)
    print("Verified and saved data/data_table.csv")
