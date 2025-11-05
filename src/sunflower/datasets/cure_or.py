from pathlib import Path

import pandas as pd

from .base import BaseDatasetInterface


def get_cure_or_class_names_from_column(class_column):
    match class_column:
        case "perspective":
            return [
                "Front (0º)",
                "Left side (90º)",
                "Back (180º)",
                "Right side (270º)",
                "Top",
            ]
        case _:
            raise NotImplementedError(
                f"Class names not available for {class_column}. Please implement!"
            )


# See https://github.com/olivesgatech/mini-CURE-OR
# See also https://zenodo.org/records/4299330
class MiniCUREORDatasetInterface(BaseDatasetInterface):
    def __init__(self, root=None, class_column=None, challenge_types=None):
        self.challenge_types = challenge_types
        super().__init__(root, class_column)

    def _setup(self) -> None:
        self.ids = []
        self.image_paths = []
        self.instance_class_names = []
        self.class_idxs = []
        self.is_training = []

        root_dir = Path(self.root)

        self.class_names = get_cure_or_class_names_from_column(self.class_column)
        i = 0
        for p, phase in enumerate(["train", "test"]):
            df = pd.read_csv(root_dir / f"{phase}.csv")
            if self.challenge_types:
                df = df[df["challengeType"].isin(self.challenge_types)]
            ids = df.imageID.to_list()
            class_idxs = [x - 1 for x in df[self.class_column].to_list()]
            self.instance_class_names += [self.class_names[x] for x in class_idxs]
            self.is_training += [phase == "train" for _ in range(len(ids))]
            self.image_paths += [Path(root_dir, phase, f"{x:05d}.jpg") for x in ids]

            self.ids += [i + x for x in range(len(ids))]  # To keep all ids unique
            self.class_idxs += class_idxs

            i += len(ids)

    def get_class_names(self):
        return self.class_names

    def get_dataset(self, phase="train", class_names=None, class_idxs=None):
        if phase not in ["train", "test"]:
            raise ValueError(
                f"{phase} is not a valid phase for the Mini CURE-OR dataset. Use either 'train' or 'test'."
            )

        if class_idxs is None and class_names is not None:
            class_idxs = []
            for cn in class_names:
                if cn not in self.class_names:
                    raise ValueError(
                        f"{cn} is not a valid class name for the CUB dataset. Please use from {self.get_class_names()}"
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
