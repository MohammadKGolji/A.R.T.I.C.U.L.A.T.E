import os
import urllib.request
import tarfile

# Make the local directory to stroe all the stuff
os.makedirs("torgo_data", exist_ok=True)

# Define TORGO stuff!
URLS = {
    "F.tar.bz2": "https://www.cs.toronto.edu/~complingweb/data/TORGO/F.tar.bz2",
    "M.tar.bz2": "https://www.cs.toronto.edu/~complingweb/data/TORGO/M.tar.bz2",
}
# Always extract archives so expected folders exist even after partial prior runs.
def extract(filepath, dest_dir):
    with tarfile.open(filepath, "r:bz2") as tar:
        tar.extractall(dest_dir)
    print(f"  Extracted to {dest_dir}/")

# Skip re-downloads on reruns to save time and bandwidth.
for filename, url in URLS.items():
    local_path = os.path.join("torgo_data", filename)

    if not os.path.exists(local_path):
        urllib.request.urlretrieve(url, local_path)
        print(f"downloaded {filename}")
    else:
        print(f"Already downloaded: {filename}")

    extract(local_path, "torgo_data")

print("\nFinsihed Finally!")
