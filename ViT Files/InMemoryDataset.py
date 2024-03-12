import torch
from torchvision import transforms
import torch.nn.functional as fun
import torchvision.transforms.functional as F
import random
from PIL import Image
import numpy as np


# ***JJ Comment: I added a variable called im_size, which is set to 224x224 in order for the Transformer pretraining code to work

class InMemoryDataset(torch.utils.data.Dataset):

    def __init__(self, data_list, preprocess_func, source='S1', select_bands=(0, 1, 2), im_size=224):
        self.data_list = data_list
        self.preprocess_func = preprocess_func
        self.source = source
        self.select_bands = select_bands
        self.im_size = im_size

    def __getitem__(self, i):
        return self.preprocess_func(self.data_list[i], self.source, self.select_bands)

    def __len__(self):
        return len(self.data_list)


def processAndAugment(data, source='S1', select_bands=(0, 1, 2), im_size=224):
    (x, y) = data
    im, label = x.copy(), y.copy()
    label = label.astype(np.float)

    if source == 'S1':
        bands = 2
    else:
        bands = len(select_bands)

    # convert to PIL for easier transforms
    ims = []
    for i in range(bands):
        ims.append(Image.fromarray(im[i]))

    label = Image.fromarray(label.squeeze())

    # Get params for random transforms
    i, j, h, w = transforms.RandomCrop.get_params(ims[0], (im_size, im_size))

    for i in range(bands):
        ims[i] = F.crop(ims[i], i, j, h, w)
    label = F.crop(label, i, j, h, w)

    if random.random() > 0.5:
        for i in range(bands):
            ims[i] = F.hflip(ims[i])
        label = F.hflip(label)

    if random.random() > 0.5:
        for i in range(bands):
            ims[i] = F.vflip(ims[i])
        label = F.vflip(label)

    if random.random() > 0.75:
        rotation = random.choice((90, 180, 270))
        for i in range(bands):
            ims[i] = F.rotate(ims[i], rotation)
        label = F.rotate(label, rotation)

    """if random.random() > 0.2:
      for i in range(bands):
        ims[i] = F.gaussian_blur(ims[i], 7)"""

    # What does this do
    if source == 'S1':
        norm = transforms.Normalize([0.6851, 0.5235], [0.0820, 0.1102])
    else:  # TODO band selector
        mean_list = np.array([0.16269160022432763, 0.13960347063125136, 0.13640611841716485,
                              0.1218228479188587, 0.14660729066303788, 0.23869029753700105, 0.284561256276994,
                              0.2622957968923778,
                              0.3077482214806557, 0.048687436781988974, 0.006377861007811543, 0.20306476302374007,
                              0.11791660722096743])
        std_list = np.array([0.07001713384623806, 0.07390945268205054, 0.07352482387959473, 0.08649366949997794,
                             0.07768803358037298, 0.09213683430927469, 0.10843734609719749, 0.10226341800670553,
                             0.1196442553176325,
                             0.03366110543131479, 0.014399923282248634, 0.09808706134697646, 0.07646083655721092])
        norm = transforms.Normalize(mean_list[np.array(select_bands)], std_list[np.array(select_bands)])

    blur = transforms.GaussianBlur(7)

    ims_T = []
    for i in range(bands):
        ims_T.append(transforms.ToTensor()(ims[i]).squeeze())

    im = torch.stack(ims_T)
    if random.random() > .2:
        im = blur(im)
    im = norm(im)

    label = transforms.ToTensor()(label).squeeze()
    if torch.sum(label.gt(.003) * label.lt(.004)):
        label *= 255
    label = label.round()

    return im, label


def processTestIm(data, source='S1', select_bands=(0, 1, 2), im_size=224):
    if source == 'S1':
        bands = 2
    else:
        bands = len(select_bands)

    (x, y) = data
    im, label = x.copy(), y.copy()
    label = label.astype(np.float)
    if source == 'S1':
        norm = transforms.Normalize([0.6851, 0.5235], [0.0820, 0.1102])
    else:  # TODO band selector
        mean_list = np.array([0.16269160022432763, 0.13960347063125136, 0.13640611841716485,
                              0.1218228479188587, 0.14660729066303788, 0.23869029753700105, 0.284561256276994,
                              0.2622957968923778,
                              0.3077482214806557, 0.048687436781988974, 0.006377861007811543, 0.20306476302374007,
                              0.11791660722096743])
        std_list = np.array([0.07001713384623806, 0.07390945268205054, 0.07352482387959473, 0.08649366949997794,
                             0.07768803358037298, 0.09213683430927469, 0.10843734609719749, 0.10226341800670553,
                             0.1196442553176325,
                             0.03366110543131479, 0.014399923282248634, 0.09808706134697646, 0.07646083655721092])
        norm = transforms.Normalize(mean_list[np.array(select_bands)], std_list[np.array(select_bands)])

    # convert to PIL for easier transforms
    im_c = []
    for i in range(bands):
        im_c.append(Image.fromarray(im[i]).resize((512, 512)))

    label = Image.fromarray(label.squeeze()).resize((512, 512))

    im_cs = []
    for i in range(bands):
        im_cs.append([F.crop(im_c[i], 0, 0, im_size, im_size), F.crop(im_c[i], 0, im_size, im_size, im_size),
                      F.crop(im_c[i], im_size, 0, im_size, im_size),
                      F.crop(im_c[i], im_size, im_size, im_size, im_size)])
    labels = [F.crop(label, 0, 0, im_size, im_size), F.crop(label, 0, im_size, im_size, im_size),
              F.crop(label, im_size, 0, im_size, im_size), F.crop(label, im_size, im_size, im_size, im_size)]

    ims = []
    for i in range(4):
        temp = []
        for j in range(bands):
            temp.append(transforms.ToTensor()(im_cs[j][i]).squeeze())
        ims.append(torch.stack(temp))

    ims = [norm(im) for im in ims]
    ims = torch.stack(ims)

    labels = [(transforms.ToTensor()(label).squeeze()) for label in labels]
    labels = torch.stack(labels)

    if torch.sum(labels.gt(.003) * labels.lt(.004)):
        labels *= 255
    labels = labels.round()

    return ims, labels