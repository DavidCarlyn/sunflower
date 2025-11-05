from pathlib import Path
from collections import defaultdict

import pandas as pd

from sunflower.utils import read_deliminted_data

from .base import BaseDatasetInterface


def extract_files(root_dir, phase="train", split=1):
    list_fp = root_dir / f"labels/{phase}{split}.txt"
    with open(list_fp, "r") as f:
        files = f.readlines()
    return files


# See https://www.robots.ox.ac.uk/~vgg/data/dtd/
class DTDDatasetInterface(BaseDatasetInterface):
    def _setup(self) -> None:
        self.ids = []
        self.image_paths = []
        self.instance_class_names = []
        self.class_idxs = []
        self.phases = []

        root_dir = Path(self.root)

        class_names = set()
        for root, dirs, files in Path(root_dir, "images").walk():
            for f in files:
                class_names.add(root.parts[-1])

        self.class_names = sorted(list(class_names))

        split_files = {}
        split_files["train"] = extract_files(root_dir, "train", 1)
        split_files["val"] = extract_files(root_dir, "val", 1)
        split_files["test"] = extract_files(root_dir, "test", 1)

        i = 0
        for p, phase in enumerate(["train", "val", "test"]):
            file_paths = split_files[phase]

            for fp in file_paths:
                self.ids.append(i)
                i += 1
                self.phases.append(phase)
                full_fp = Path(root_dir, "images", fp.strip())
                self.image_paths.append(full_fp)
                cn = full_fp.parts[-2]
                assert cn in self.class_names
                self.instance_class_names.append(cn)
                self.class_idxs.append(self.class_names.index(cn))

    def get_class_names(self):
        return self.class_names

    def get_dataset(self, phase="train", class_names=None, class_idxs=None):
        if phase not in ["train", "val", "test"]:
            raise ValueError(
                f"{phase} is not a valid phase for the DTD dataset. Use either 'train', 'val' or 'test'."
            )

        if class_idxs is None and class_names is not None:
            class_idxs = []
            for cn in class_names:
                if cn not in self.class_names:
                    raise ValueError(
                        f"{cn} is not a valid class name for the DTD dataset. Please use from {self.get_class_names()}"
                    )
                class_idxs.append(self.class_names.index(cn))

        if class_idxs is not None:
            min_cls_idx = min(class_idxs)
            max_cls_idx = max(class_idxs)
            if min_cls_idx < 0:
                raise ValueError(f"{min_cls_idx} is not a valid class index")
            if max_cls_idx > max(self.class_idxs):
                raise ValueError(f"{max_cls_idx} is not a valid class index")

        filtered_ids = []
        filtered_paths = []
        filtered_class_names = []
        filtered_class_idxs = []
        for id, path, cn, cls_idx, ist_phase in zip(
            self.ids,
            self.image_paths,
            self.instance_class_names,
            self.class_idxs,
            self.phases,
        ):
            # Filter by phase
            if phase != ist_phase:
                continue

            # Filter by class idx
            if class_idxs is not None and cls_idx not in class_idxs:
                continue

            filtered_ids.append(id)
            filtered_paths.append(path)
            filtered_class_names.append(cn)
            filtered_class_idxs.append(cls_idx)

        return {
            "ids": filtered_ids,
            "paths": filtered_paths,
            "class_names": filtered_class_names,
            "labels": filtered_class_idxs,
        }
