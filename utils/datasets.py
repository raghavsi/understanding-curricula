import os.path
import pickle
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
from PIL import Image

from torchvision.datasets.utils import check_integrity, download_and_extract_archive, verify_str_arg
from torchvision.datasets.vision import VisionDataset
from pathlib import Path


from PIL import Image

from torchvision.datasets.folder import find_classes, make_dataset



class CIFAR10T(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    base_folder = "cifar-10-batches-py"
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = "c58f30108f718f92721af3b95e74349a"
    train_list = [
        ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
        ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
        ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
        ["data_batch_4", "634d18415352ddfa80567beed471001a"],
        #["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
    ]

    test_list = [
        ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
        #["test_batch", "40351d587109b95175f43aff81a1287e"],
    ]
    meta = {
        "filename": "batches.meta",
        "key": "label_names",
        "md5": "5ff9c542aee3614f3951f8cda6e48888",
    }

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                if "labels" in entry:
                    self.targets.extend(entry["labels"])
                else:
                    self.targets.extend(entry["fine_labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        if not check_integrity(path, self.meta["md5"]):
            raise RuntimeError("Dataset metadata file not found or corrupted. You can use download=True to download it")
        with open(path, "rb") as infile:
            data = pickle.load(infile, encoding="latin1")
            self.classes = data[self.meta["key"]]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target#, index

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        for filename, md5 in self.train_list + self.test_list:
            fpath = os.path.join(self.root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"


class CIFAR100T(CIFAR10T):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """

    base_folder = "cifar-100-python"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list = [
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
    ]

    test_list = [
        ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
    ]
    meta = {
        "filename": "meta",
        "key": "fine_label_names",
        "md5": "7973b15100ade9c7d40fb424638fde48",
    }
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        #if self.train :
        downloaded_list = self.train_list
        #else:
        #    downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                if "labels" in entry:
                    self.targets.extend(entry["labels"])
                else:
                    self.targets.extend(entry["fine_labels"])
        if self.train:
            self.data = self.data[0][:40000,:]
            self.targets = self.targets[:40000]
        else:
            self.data = self.data[0][40000:,:]
            self.targets = self.targets[40000:]
        
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        print(self.data.shape,len(self.targets))
        self._load_meta()



class Imagenette(VisionDataset):
    """`Imagenette <https://github.com/fastai/imagenette#imagenette-1>`_ image classification dataset.

    Args:
        root (str or ``pathlib.Path``): Root directory of the Imagenette dataset.
        split (string, optional): The dataset split. Supports ``"train"`` (default), and ``"val"``.
        size (string, optional): The image size. Supports ``"full"`` (default), ``"320px"``, and ``"160px"``.
        download (bool, optional): If ``True``, downloads the dataset components and places them in ``root``. Already
            downloaded archives are not downloaded again.
        transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed
            version, e.g. ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.

     Attributes:
        classes (list): List of the class name tuples.
        class_to_idx (dict): Dict with items (class name, class index).
        wnids (list): List of the WordNet IDs.
        wnid_to_idx (dict): Dict with items (WordNet ID, class index).
    """

    _ARCHIVES = {
        "full": ("https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz", "fe2fc210e6bb7c5664d602c3cd71e612"),
        "320px": ("https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz", "3df6f0d01a2c9592104656642f5e78a3"),
        "160px": ("https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz", "e793b78cc4c9e9a4ccc0c1155377a412"),
    }
    _WNID_TO_CLASS = {
        "n01440764": ("tench", "Tinca tinca"),
        "n02102040": ("English springer", "English springer spaniel"),
        "n02979186": ("cassette player",),
        "n03000684": ("chain saw", "chainsaw"),
        "n03028079": ("church", "church building"),
        "n03394916": ("French horn", "horn"),
        "n03417042": ("garbage truck", "dustcart"),
        "n03425413": ("gas pump", "gasoline pump", "petrol pump", "island dispenser"),
        "n03445777": ("golf ball",),
        "n03888257": ("parachute", "chute"),
        "n02086240": ("Australian terrier"),
        "n02087394": ("Border terrier"),
	"n02088364": ("Samoyed"),
	"n02089973": ("Shih-Tzu"),
	"n02093754": ("English foxhound"),
	"n02096294": ("Rhodesian ridgeback"),
	"n02099601": ("Dingo"),
	"n02105641": ("Golden retriever"),
        "n02111889": ("Old English sheepdog"),
        "n02115641": ("Beagle")
        
    }

    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        size: str = "full",
        download=False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)

        self._split = verify_str_arg(split, "split", ["train", "val"])
        self._size = verify_str_arg(size, "size", ["full", "320px", "160px"])

        self._url, self._md5 = self._ARCHIVES[self._size]
        self._size_root = Path(self.root) / Path(self._url).stem
        self._size_root = Path(self.root) / Path("ImagenetWoofEtte")
        self._image_root = str(self._size_root / self._split)
        #print('aaaa',self._url, self._size_root, self._image_root)
        if download:
            self._download()
        elif not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it.")

        self.wnids, self.wnid_to_idx = find_classes(self._image_root)
        self.classes = [self._WNID_TO_CLASS[wnid] for wnid in self.wnids]
        self.class_to_idx = {
            class_name: idx for wnid, idx in self.wnid_to_idx.items() for class_name in self._WNID_TO_CLASS[wnid]
        }
        self._samples = make_dataset(self._image_root, self.wnid_to_idx, extensions=".jpeg")
        self.targets = []
        self.paths = []
        for s in self._samples:
            self.targets.append(s[1])
            self.paths.append(s[0].split('/')[-1].encode('UTF-8'))

    def _check_exists(self) -> bool:
        return self._size_root.exists()

    def _download(self):
        if self._check_exists():
            raise RuntimeError(
                f"The directory {self._size_root} already exists. "
                f"If you want to re-download or re-extract the images, delete the directory."
            )

        download_and_extract_archive(self._url, self.root, md5=self._md5)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        path, label = self._samples[idx]
        image = Image.open(path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __len__(self) -> int:
        return len(self._samples)
