from pathlib import Path

import pandas as pd

from .base import BaseDatasetInterface


class WaterBirdDatasetInterface(BaseDatasetInterface):
    def _setup(self) -> None:
        if self.class_column is None:
            self.class_column = "background"

        root_dir = Path(self.root)

        df = pd.read_csv(root_dir / "metadata.csv")
        self.ids = df.img_id.tolist()
        if self.class_column == "background":
            self.class_idxs = df.place.tolist()
        elif self.class_column == "bird":
            self.class_idxs = df.y.tolist()
        elif self.class_column == "background-fg":
            # TODO
            raise NotImplementedError(
                "Please implement for class column 'background-fg'"
            )
        elif self.class_column == "species":
            self.class_idxs = df.img_filename.str.split(".").str[0].astype(int) - 1

        self.image_paths = [root_dir / x for x in df.img_filename.tolist()]

        phase_map = {0: "train", 1: "val", 2: "test"}
        self.phases = [phase_map[x] for x in df.split]

        if self.class_column in ["background", "bird"]:
            self.class_names = ["land", "water"]
            self.instance_class_names = [self.class_names[x] for x in self.class_idxs]
        elif self.class_column == "background-fg":
            # TODO
            raise NotImplementedError(
                "Please implement for class column 'background-fg'"
            )
        elif self.class_column == "species":
            self.instance_class_names = (
                df.img_filename.str.split(".").str[1].apply(lambda x: x.split("/")[0])
            )

            lbl_to_class_map = {}
            for lbl, class_name in zip(self.class_idxs, self.instance_class_names):
                lbl_to_class_map[lbl] = class_name

            self.class_names = [lbl_to_class_map[i] for i in range(200)]

    def get_class_names(self):
        return self.class_names

    def get_dataset(self, phase="train", class_names=None, class_idxs=None):
        if phase not in ["train", "val", "test"]:
            raise ValueError(
                f"{phase} is not a valid phase for the Waterbird dataset. Use either 'train', 'val' or 'test'."
            )

        if class_idxs is None and class_names is not None:
            class_idxs = []
            for cn in class_names:
                if cn not in self.class_names:
                    raise ValueError(
                        f"{cn} is not a valid class name for the Waterbird dataset. Please use from {self.get_class_names()}"
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
