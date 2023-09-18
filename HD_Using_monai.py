import SimpleITK as sitk
from monai.metrics import compute_hausdorff_distance
import numpy as np

y_pred  = sitk.ReadImage(r"D:\PROJ_WORK\labelsTr\001_LA_ED.nii.gz")
y_pred = sitk.GetArrayFromImage(y_pred)

gt  = sitk.ReadImage(r"D:\PROJ_WORK\labelsTr\001_LA_ED.nii.gz")
gt = sitk.GetArrayFromImage(gt)

pre_lv = np.zeros(gt.shape)
gt_lv = np.zeros(gt.shape)

 
pre_lv[np.where(y_pred==1)] = 1
gt_lv[np.where(gt==1)] = 1

    
pre_lv = np.expand_dims(pre_lv, axis=2)
gt_lv = np.expand_dims(gt_lv, axis=2)


hd = compute_hausdorff_distance(pre_lv, gt_lv)
