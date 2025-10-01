import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from .utils.matrix import square_over_bg_falloff


class White_Square_dataset(Dataset):
    """Dataset of white squares on black background"""

    def __init__(
            self,
            img_size=64,
            square_size=2) -> None:
        super().__init__()

        self.img_size = img_size
        self.square_size = square_size
        self.df = None
        self.generate_dataset()

    def generate_dataset(self):
        # generate all possible x and y coordinates
        x = np.arange(0, self.img_size - self.square_size)
        y = np.arange(0, self.img_size - self.square_size)
        # create the grid
        xx, yy = np.meshgrid(x, y)
        # generate all possible combinations
        all_combinations = np.vstack([xx.ravel(), yy.ravel()]).T
        # create the dataframe
        self.df = pd.DataFrame(all_combinations, columns=["x", "y"])
        print(f"Generated dataset with {len(self.df)} samples")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get the row
        row = self.df.iloc[idx]
        # get the x and y coordinates
        x = row.x
        y = row.y
        # create the image
        img = square_over_bg_falloff(x, y, self.img_size, self.square_size)
        # add a channel dimension
        img = img.unsqueeze(0)

        # get another random row
        row = self.df.sample().iloc[0]
        # get the x and y coordinates
        x = row.x
        y = row.y
        # create the image
        img2 = square_over_bg_falloff(x, y, self.img_size, self.square_size)
        # add a channel dimension
        img2 = img2.unsqueeze(0)

        # return the images
        return img, img2