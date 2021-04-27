import unittest
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import DataLoader
from data.dataset import AnimeFacesDataset


plt.rcParams["figure.figsize"] = (8, 8)
plt.gca().set_axis_off()
plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                    hspace=0, wspace=0)
plt.margins(0, 0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())


class TestDataset(unittest.TestCase):

    def setUp(self):
        dataset = AnimeFacesDataset('path to dataset')
        self.train_dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

    def test_batch(self):
        batch = next(iter(self.train_dataloader))
        img_grid = torchvision.utils.make_grid(batch, normalize=True)
        plt.imshow(img_grid.permute(1, 2, 0))
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    unittest.main()
