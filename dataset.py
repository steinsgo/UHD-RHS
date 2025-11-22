
import os
import random

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CloudRemovalDataset(Dataset):
    """Dataset loader for the 8KDehaze dataset.

    The dataset contains a ``clear`` folder and four folders with different
    haze levels named ``cloud_L1`` to ``cloud_L4``. For each index, a random
    haze level is chosen so the network sees diverse degradation levels.
    """

    def __init__(self, root_dir, normalize, crop_size):
        self.root_dir = root_dir
        self.clear_dir = os.path.join(self.root_dir, 'clear')
        self.image_filenames = [
            img for img in os.listdir(self.clear_dir)
            if img.endswith('.png') or img.endswith('.jpg')
        ]
        transforms_list = [transforms.ToTensor()]
        self.crop_size = crop_size
        if normalize:
            transforms_list.append(
                transforms.Normalize(
                    [0.45837133, 0.47633536, 0.44432645],
                    [0.16936361, 0.15927625, 0.15468806],
                )
            )
        self.transform = transforms.Compose(transforms_list)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        cloud_dir = os.path.join(
            self.root_dir, f'cloud_L{random.choice([1, 2, 3, 4])}'
        )
        cloud_img_name = os.path.join(cloud_dir, self.image_filenames[idx])
        clear_img_name = os.path.join(self.clear_dir, self.image_filenames[idx])

        cloud_img = Image.open(cloud_img_name).convert("RGB")
        clear_img = Image.open(clear_img_name).convert("RGB")

        if cloud_img.size != clear_img.size:
            raise ValueError("The size of the input image pairs is inconsistent")

        if self.crop_size is not None and self.crop_size > 0:
            if (
                self.crop_size > clear_img.size[0]
                or self.crop_size > clear_img.size[1]
            ):
                raise ValueError("Crop size is larger than the image size")
            start_x = random.randint(0, clear_img.size[0] - self.crop_size)
            start_y = random.randint(0, clear_img.size[1] - self.crop_size)
            clear_img = clear_img.crop(
                (start_x, start_y, start_x + self.crop_size, start_y + self.crop_size)
            )
            cloud_img = cloud_img.crop(
                (start_x, start_y, start_x + self.crop_size, start_y + self.crop_size)
            )

        random_rotate = random.choice([0, 90, 180, 270])
        clear_img = clear_img.rotate(random_rotate)
        cloud_img = cloud_img.rotate(random_rotate)
        clear_img = self.transform(clear_img)
        cloud_img = self.transform(cloud_img)
        sample = {'cloud_img': cloud_img, 'clear_img': clear_img}
        return sample
    

# 新增：支持 clear/addrain 结构的去雨数据集
class RainRemovalDataset(Dataset):
    """Dataset loader for paired clear/addrain rain removal dataset."""
    def __init__(self, root_dir, normalize, crop_size):
        self.root_dir = root_dir
        self.clear_dir = os.path.join(self.root_dir, 'gt')
        self.rain_dir = os.path.join(self.root_dir, 'input')
        self.image_filenames = [
            img for img in os.listdir(self.clear_dir)
            if img.lower().endswith('.png') or img.lower().endswith('.jpg')
        ]
        transforms_list = [transforms.ToTensor()]
        self.crop_size = crop_size
        if normalize:
            transforms_list.append(
                transforms.Normalize(
                    [0.45837133, 0.47633536, 0.44432645],
                    [0.16936361, 0.15927625, 0.15468806],
                )
            )
        self.transform = transforms.Compose(transforms_list)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        rain_img_name = os.path.join(self.rain_dir, self.image_filenames[idx])
        clear_img_name = os.path.join(self.clear_dir, self.image_filenames[idx])
        rain_img = Image.open(rain_img_name).convert("RGB")
        clear_img = Image.open(clear_img_name).convert("RGB")

        if rain_img.size != clear_img.size:
            raise ValueError("The size of the input image pairs is inconsistent")

        if self.crop_size is not None and self.crop_size > 0:
            if (
                self.crop_size > clear_img.size[0]
                or self.crop_size > clear_img.size[1]
            ):
                raise ValueError("Crop size is larger than the image size")
            start_x = random.randint(0, clear_img.size[0] - self.crop_size)
            start_y = random.randint(0, clear_img.size[1] - self.crop_size)
            clear_img = clear_img.crop(
                (start_x, start_y, start_x + self.crop_size, start_y + self.crop_size)
            )
            rain_img = rain_img.crop(
                (start_x, start_y, start_x + self.crop_size, start_y + self.crop_size)
            )

        random_rotate = random.choice([0, 90, 180, 270])
        clear_img = clear_img.rotate(random_rotate)
        rain_img = rain_img.rotate(random_rotate)
        clear_img = self.transform(clear_img)
        rain_img = self.transform(rain_img)
        sample = {'cloud_img': rain_img, 'clear_img': clear_img}
        return sample

if __name__ == '__main__':
    #dataset = CloudRemovalDataset(r"./datasets/8KDehaze_mini", False, crop_size=2048)
    dataset = RainRemovalDataset(r"./datasets/your_rain_dataset", False, crop_size=512)
    print(dataset[0])