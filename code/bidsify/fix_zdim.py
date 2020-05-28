import nibabel as nib
import numpy as np
from tqdm import tqdm
from nilearn import image
from glob import glob

files = sorted(glob('../sub*/ses*/func/*bold.nii.gz'))
for f in tqdm(files):
    f_img = nib.load(f)
    zdim = f_img.header['pixdim'][3]
    if (zdim % 2.97) > 0.0000001:
        print(f)
        diff = 2.97 - zdim
        aff = f_img.affine.copy()
        aff[2, 2] += diff
        print(f_img.affine)
        print(f_img.header['pixdim'])
        tmp = image.resample_img(f_img, aff)
        print(tmp.affine)
        print(tmp.header['pixdim'])
        tmp.to_filename(f.replace('.nii.gz', '_fixed.nii.gz'))

