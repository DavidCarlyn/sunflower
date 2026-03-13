import os
from pathlib import Path

import pandas as pd

from .base import BaseDatasetInterface


class ImageNetEDatasetInterface(BaseDatasetInterface):
    def __init__(self, root=None, augment="ori", keep_imagenet_label_index=False):
        self.keep_imagenet_label_index = keep_imagenet_label_index
        self.augment = augment
        super().__init__(root)

    def _setup(self) -> None:
        self.ids = []
        self.image_paths = []
        self.instance_class_names = []
        self.class_idxs = []
        self.is_training = []

        root_dir = Path(self.root)

        label_df = pd.read_csv(
            root_dir / "labels.txt", sep="\t", names=["path", "label"]
        )
        label_df.path = [p.split(".")[0] for p in label_df.path]
        path_label_map = dict(zip(label_df.path.tolist(), label_df.label.tolist()))

        self.class_names = sorted(label_df.label.unique().tolist())

        id = 0
        for phase, dir_name in zip(["train", "test"], ["ori", self.augment]):
            paths = os.listdir(root_dir / dir_name)
            for path in paths:
                self.ids.append(id)
                id += 1
                self.image_paths.append(Path(root_dir, dir_name, path))
                path_name = path.split(".")[0]
                label = path_label_map[path_name]
                self.instance_class_names.append(
                    str(label)
                )  # TODO link this to imagenet labels later
                if self.keep_imagenet_label_index:
                    self.class_idxs.append(label)
                else:
                    self.class_idxs.append(self.class_names.index(label))
                self.is_training.append(phase == "train")

    def get_class_names(self):
        return self.class_names

    def get_dataset(self, phase="train", class_names=None, class_idxs=None):
        if phase not in ["train", "test"]:
            raise ValueError(
                f"{phase} is not a valid phase for the Imagenet-E dataset. Use either 'train' or 'test'."
            )

        if class_idxs is None and class_names is not None:
            class_idxs = []
            for cn in class_names:
                if cn not in self.class_names:
                    raise ValueError(
                        f"{cn} is not a valid class name for the Imagenet-E dataset. Please use from {self.get_class_names()}"
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
        for id, path, cn, cls_idx, is_train in zip(
            self.ids,
            self.image_paths,
            self.instance_class_names,
            self.class_idxs,
            self.is_training,
        ):
            # Filter by phase
            if phase == "train" and int(is_train) != 1:
                continue
            if phase == "test" and int(is_train) != 0:
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
