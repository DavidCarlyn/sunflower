from pathlib import Path

import pandas as pd

from .base import BaseDatasetInterface


# See https://github.com/olivesgatech/mini-CURE-OR
# See also https://zenodo.org/records/4299330
class EgoOrientDatasetInterface(BaseDatasetInterface):
    def __init__(self, root=None):
        super().__init__(root)

    def _setup(self) -> None:
        self.ids = []
        self.image_paths = []
        self.instance_class_names = []
        self.class_idxs = []
        self.is_training = []

        root_dir = Path(self.root)
        train_df = pd.read_json(root_dir / "imagenet_train.json")
        train_df = train_df.drop_duplicates("image")
        train_paths = train_df.image.tolist()
        train_labels = train_df.direction.tolist()

        test_df = pd.read_json(root_dir / "benchmark.json")
        test_df = test_df.drop_duplicates("image")
        test_paths = [Path(p).parts[-1] for p in test_df.image.tolist()]
        test_labels = test_df.original_label.tolist()

        self.class_names = sorted(train_df.direction.unique().tolist())

        id = 0
        for phase, paths, labels in zip(
            ["train", "test"],
            [train_paths, test_paths],
            [train_labels, test_labels],
        ):
            for path, lbl in zip(paths, labels):
                self.ids.append(id)
                id += 1
                self.image_paths.append(Path(root_dir, "images", path))
                self.instance_class_names.append(lbl)
                self.class_idxs.append(self.class_names.index(lbl))
                self.is_training.append(phase == "train")

    def get_class_names(self):
        return self.class_names

    def get_dataset(self, phase="train", class_names=None, class_idxs=None):
        if phase not in ["train", "test"]:
            raise ValueError(
                f"{phase} is not a valid phase for the EgoOrient dataset. Use either 'train' or 'test'."
            )

        if class_idxs is None and class_names is not None:
            class_idxs = []
            for cn in class_names:
                if cn not in self.class_names:
                    raise ValueError(
                        f"{cn} is not a valid class name for the EgoOrient dataset. Please use from {self.get_class_names()}"
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
