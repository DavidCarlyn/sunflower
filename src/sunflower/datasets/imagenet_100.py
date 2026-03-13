from pathlib import Path

from sunflower.utils import read_deliminted_data

from .base import BaseDatasetInterface


class ImageNet100Interface(BaseDatasetInterface):
    def _setup(self) -> None:
        self.ids = []
        self.image_paths = []
        self.instance_class_names = []
        self.class_idxs = []
        self.is_training = []

        root_dir = Path(self.root)

        def create_two_column_map(fp):
            return {key: value for key, value in read_deliminted_data(fp, delim=" ")}

        train_path_to_label_map = create_two_column_map(root_dir / "train.txt")
        val_path_to_label_map = create_two_column_map(root_dir / "val.txt")

        label_to_classname_map = {}
        for path, label in train_path_to_label_map.items():
            label = int(label)
            class_name = path.split("/")[-2]
            label_to_classname_map[label] = class_name

        assert len(label_to_classname_map.keys()) == 1000

        self.class_names = []
        for i in range(1000):
            self.class_names.append(label_to_classname_map[i])

        id = 0
        for phase, path_to_label_path in zip(
            ["train", "test"], [train_path_to_label_map, val_path_to_label_map]
        ):
            for local_path, label in path_to_label_path.items():
                self.ids.append(id)
                id += 1
                self.image_paths.append(Path(root_dir, local_path))
                cls_idx = int(label)
                cls_name = label_to_classname_map[cls_idx]
                self.class_idxs.append(cls_idx)
                self.instance_class_names.append(cls_name)

                self.is_training.append(phase == "train")

    def get_class_names(self):
        return self.class_names

    def get_dataset(self, phase="train", class_names=None, class_idxs=None):
        if phase not in ["train", "test"]:
            raise ValueError(
                f"{phase} is not a valid phase for the ImageNet100 dataset. Use either 'train' or 'test'."
            )

        if class_idxs is None and class_names is not None:
            class_idxs = []
            for cn in class_names:
                if cn not in self.class_names:
                    raise ValueError(
                        f"{cn} is not a valid class name for the ImageNet100 dataset. Please use from {self.get_class_names()}"
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
