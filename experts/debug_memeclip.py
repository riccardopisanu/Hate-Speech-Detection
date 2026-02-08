import os
import torch
import pandas as pd
import argparse
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import clip


class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def get_dummy_config():
    return DotDict(
        {
            "model": DotDict({"clip_model": "ViT-L/14", "class_names": ["no", "yes"]}),
            "data": DotDict({"num_classes": 2}),
            "map_dim": 1024,
        }
    )


class CosineClassifier(nn.Module):
    def __init__(self, feat_dim, num_classes=2):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_classes, feat_dim))


class MemeCLIP_Skeleton(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = CosineClassifier(1024, 2)
        self.image_map = nn.Sequential(nn.Linear(10, 10))
        self.text_map = nn.Sequential(nn.Linear(10, 10))
        self.pre_output = nn.Sequential(nn.Linear(10, 10))


def check_data(test_file, image_root):

    if not os.path.exists(test_file):
        return

    try:
        if test_file.endswith(".jsonl"):
            df = pd.read_json(test_file, lines=True)
        else:
            try:
                df = pd.read_csv(test_file, sep=",")
            except:
                df = pd.read_csv(test_file, sep="\t")
    except Exception as e:
        return

    cols = {c.lower().strip(): c for c in df.columns}
    img_col = next((cols[c] for c in ["img", "image_name", "id"] if c in cols), None)

    if not img_col:
        return

    first_img_name = str(df.iloc[0][img_col])

    path_1 = os.path.join(image_root, os.path.basename(first_img_name))
    path_2 = os.path.join(image_root, first_img_name)

    if os.path.exists(path_1):
        print("    FILE TROVATO!")
    else:
        print("    FILE NON TROVATO")

    if os.path.exists(path_2):
        print("    FILE TROVATO!")
    else:
        print("    FILE NON TROVATO")

    if not os.path.exists(path_1) and not os.path.exists(path_2):
        try:
            pass
        except:
            pass


def check_weights(checkpoint_path):

    if not os.path.exists(checkpoint_path):
        return

    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt["state_dict"]

        keys = list(state_dict.keys())
        for k in keys[:10]:
            pass

        class_keys = [k for k in keys if "classifier" in k]
        if class_keys:
            for k in class_keys:
                print(f"   FOUND: {k}  <-- IMPORTANTE")
        else:
            pass

        map_keys = [k for k in keys if "map" in k][:5]
        for k in map_keys:
            print(f"   FOUND: {k}")

    except Exception as e:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="")
    parser.add_argument("--model_type", type=str, default="")
    args = parser.parse_args()

    check_data(args.test_file, args.image_root)
    check_weights(args.checkpoint_path)
