import os
import sys
import os.path as op
from glob import glob
from zipfile import ZipFile
from urllib.request import urlopen

data_dir = op.join('..', 'stimuli', 'london_face_dataset')
if not op.isdir(data_dir):
    os.makedirs(data_dir, exist_ok=True)

f_out = op.join(data_dir, 'london_face_dataset.zip')
if op.isfile(op.join(data_dir, 'facelab_london.pdf')):
    print("Data was already downloaded!")
else:
    print("Downloading London Face dataset")
    URL = 'https://ndownloader.figshare.com/articles/5047666/versions/3'
    f = urlopen(URL)

    # Open our local file for writing
    with open(f_out, "wb") as local_file:
        local_file.write(f.read())

    with ZipFile(f_out, 'r') as zip_ref:
        zip_ref.extractall(data_dir)

    os.remove(f_out)

zips = sorted(glob(op.join(data_dir, '*.zip')))
for zp in zips:
    print("Unzipping %s" % op.basename(zp))
    with ZipFile(zp, 'r') as zip_ref:
        zip_ref.extractall(data_dir)

    os.remove(zp)
