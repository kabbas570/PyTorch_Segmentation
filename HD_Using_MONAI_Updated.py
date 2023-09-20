from monai.transforms import LoadImaged, Compose, AddChanneld
from monai.metrics import HausdorffDistanceMetric
import torch

data_dict = {"image": "001_LA_ED.nii.gz"}

tx = Compose(
    [
        LoadImaged(keys=["image"]),
        AddChanneld(keys=["image"]),
    ]
)

data = tx(data_dict)

gtlabel_pt = torch.nn.functional.one_hot((data["image"].unsqueeze(0)).to(torch.long), num_classes=4,)

hdmetric = HausdorffDistanceMetric(
    include_background=False,
    distance_metric="euclidean",
    percentile=95.0,
)

hd95 = hdmetric(y_pred=gtlabel_pt, y=gtlabel_pt)

print(hd95)
