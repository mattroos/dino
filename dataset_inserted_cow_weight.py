import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
import random
import os
from PIL import Image
import pandas as pd

## Just for development
import pdb
import time
import matplotlib.pyplot as plt
plt.ion()


class ObjectsAndBackgroundsDataset(Dataset):
    '''
    Images of segemented objects paired with background images
    into which they can be inserted.
    '''

    def __init__(self, path_objects, path_backgrounds, weight_csv, pattern_objects=None,
                 pattern_backgrounds=None, transform=None, shrinkages_per_object=1):
        random.seed(0)

        # Any transforms will be applied to object and background images independently
        self.transform = transform

        self.shrinkages_per_object = shrinkages_per_object

        # Get list of object files that contain pattern
        self.path_objects = path_objects
        self.pattern_objects = pattern_objects
        self.filenames_objects = self._get_filename_list(path_objects, pattern_objects)
        self.n_objects = len(self.filenames_objects)

        # Get list of background files that contain pattern
        self.path_backgrounds = path_backgrounds
        self.pattern_backgrounds = pattern_backgrounds
        self.filenames_backgrounds = self._get_filename_list(path_backgrounds, pattern_backgrounds)
        self.n_backgrounds = len(self.filenames_backgrounds)

        # Get object weights from csv file
        self.weights = self._get_weights(weight_csv)

    def _get_filename_list(self, path, pattern):
        filenames = os.listdir(path)
        if pattern:
            filenames = [os.path.join(path, f) for f in filenames if pattern in f]
        else:
            filenames = [os.path.join(path, f) for f in filenames]
        random.shuffle(filenames)
        return filenames

    def __len__(self):
        '''
        Use the number of object images as the length. The background images is always
        randomly chosen.
        If using a fixed list of shrinkage sizes, then the length is the number of
        object images times the number of shrinkage sizes.
        '''
        return self.n_objects * self.shrinkages_per_object

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        idx_obj = idx//self.shrinkages_per_object  # Repeat an object image shrinkages_per_object times
        idx_bg = np.random.randint(0, high=self.n_backgrounds)

        # Get object image
        im_obj = Image.open(self.filenames_objects[idx_obj])

        # Load object segmentation mask
        np_filename = self.filenames_objects[idx_obj][:-5]  + '_mask.npy'
        mask = np.load(np_filename)

        # Get background image
        im_bg = Image.open(self.filenames_backgrounds[idx_bg])

        try:
            sample = {'image_object':im_obj, 'mask_object':mask, 'image_background':im_bg, 'weight':self.weights[idx_obj]}
        except:
            pdb.set_trace()

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _get_max_object_length(self):
        mx = 0
        for fn in self.filenames_objects:
            im_obj = Image.open(fn)
            mx = max(max(im_obj.height, im_obj.width), mx)

        return mx

    def _get_weights(self, csv_filename):
        df = pd.read_csv(csv_filename, sep=',', header=0)
        weights = []
        for fn in self.filenames_objects:
            head, tail = os.path.split(fn)
            fn_video = '_'.join(tail.split('_')[:3]) + '.mp4'
            # This is ugly
            weights.append(np.array(df.loc[(df['filename']==fn_video)]['weight_20210223'])[0])
        return weights


class IndependentColorJitter():
    '''
    Independently jitter the color of the object and background images.
    '''
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.color_transform = transforms.ColorJitter(brightness=brightness,
                                                      contrast=contrast,
                                                      saturation=saturation,
                                                      hue=hue)

    def __call__(self, sample):
        sample['image_background'] = self.color_transform(sample['image_background'])
        
        sample['image_object'] = self.color_transform(sample['image_object'])
        # For pixels that weren't part of the object, need to reset them to all zeros/black
        # so jitter doesn't "create" new pixels as part of the segmented object. Actually
        # not certain this is needed. Can ColorJitter ever change (0,0,0) pixels to
        # something else?
        x = np.array(sample['image_object'])
        x[np.logical_not(sample['mask_object'])] = np.array([[[0, 0, 0]]])
        sample['image_object'] = Image.fromarray(x)

        return sample


class InsertObjectInBackground():
    '''
    For an input dict containing 'image_object' and 'image_background'...
      1. Extract a random crop from the background image of specified output size.
      2. Resize the object image to a random size - one that will fit in the
         background crop.
      3. Insert the object image into the background image at a random location,
         but do so only for the pixels with intensity > 0. Pixels with an
         intensity of 0 are assumed to not belong to the segmented object.
      4. Return the new image along with (1) the number of object pixels in the
         original object image and (2) the number of object pixels in the
         resized object image.
    '''
    MEAN = torch.Tensor([[[0.485, 0.456, 0.406]]])  # ImageNet channel means, RGB
    STD = torch.Tensor([[[0.229, 0.224, 0.225]]])   # ImageNet channel standard deviations, RGB

    def __init__(self, output_size, max_object_length, object_shrinkage, random_flip=True, channels_first=True, normalize=True):
        '''
        output_size: int indicating output image size (3, output_size, output_size)
                     where 3 is the number of channels.

        object_shrinkage: Factor indicating how small the object may be resized
                          to, relative to the output_size. E.g., if output_size
                          is 1000 and object_shrinkage is 0.5 then the largest
                          dimension of the object will be shrunk to between
                          0.5*1000==500 and 1000.

                          OR: If object_shrinkage is an iterable, cycle through the
                          values, using them as the shrinkage factor rather than
                          a random value.
        '''
        self.output_size = output_size
        self.max_object_length = max_object_length
        self.object_shrinkage = object_shrinkage
        self.random_flip = random_flip
        self.channels_first = channels_first
        self.normalize = normalize
        self.crop_transform = transforms.RandomResizedCrop(output_size, scale=(0.5, 1.0), ratio=(0.75, 1.33))
        self.shrink_counter = 0  # to keep track of which shrinkage to use, if object_shrinkage is an iterable
        if hasattr(self.object_shrinkage, '__iter__'):
            self.n_shrinks = len(self.object_shrinkage)

    def __call__(self, sample):
        bg = self.crop_transform(sample['image_background'])
        area_original = np.sum(sample['mask_object'])
        
        # Determine the new size of the object image, and an estimate weight for that resizing
        if hasattr(self.object_shrinkage, '__iter__'):
            f_shrink = self.object_shrinkage[self.shrink_counter%self.n_shrinks]
            self.shrink_counter += 1
        else:
            f_shrink = np.random.uniform(self.object_shrinkage, 1.0)
        # All images are scaled down to self.output_size/self.max_object_length, and
        # then further resized by f_shrink...
        f_resize = self.output_size/self.max_object_length * f_shrink
        weight_new = sample['weight'] * f_shrink**3
        h_orig = sample['image_object'].height
        w_orig = sample['image_object'].width
        h_new = round(h_orig * f_resize)
        w_new = round(w_orig * f_resize)

        # Resize the object
        obj = sample['image_object'].resize((w_new, h_new), resample=Image.BILINEAR)

        # Resize mask and convert to uints of 0 and 255 in order to use PIL to resize.
        mask = Image.fromarray((sample['mask_object']*255).astype(np.uint8))
        mask = mask.resize((w_new, h_new), resample=Image.BILINEAR)
        mask = np.array(mask) > 0

        # Insert masked object into random location in background
        top = np.random.randint(0, high=self.output_size-h_new+1)
        left = np.random.randint(0, high=self.output_size-w_new+1)
        insert = np.zeros((self.output_size, self.output_size, 3), dtype=np.uint8)
        insert[top:top+h_new, left:left+w_new, :] = obj
        mask_insert = np.full((self.output_size, self.output_size), False)
        mask_insert[top:top+h_new, left:left+w_new] = mask
        mask_insert = np.repeat(mask_insert[:,:,None], 3, axis=2)
        im_new = np.where(mask_insert, insert, bg)

        # When batching with dataloader, numpy operations may fail, so converting to torch
        im_new = torch.tensor(im_new)

        # Randomly flip, horizontally
        if self.random_flip:
            if np.random.choice((True, False)):
                im_new = torch.flip(im_new, (1,))

        # Normalize
        if self.normalize:
            im_new = (im_new/255.0 - self.MEAN) / self.STD

        # Channels first
        if self.channels_first:
            im_new = im_new.permute(2, 0, 1)

        # return {'image':im_new, 'weight':weight_new}
        return im_new, weight_new


if __name__ == "__main__":

    PATH_BACKGROUNDS = '/Data/DairyTech/Flickr_fields_train_val_sets/train/dummy'
    COW_IMAGE_FILE_ENDSWITH = 'cow_side_seg.tiff'
    WEIGHT_CSV = '/home/mroos/Data/DairyTech/video_info_2021-Feb-Mar.csv'

    # Create both a train and val dataset, and then determine the maximum cow image
    # length across both datasets.
    PATH_COWS = '/home/mroos/Data/DairyTech/labelme/scaled_segmented_train_val_sets/train/dummy'
    dataset = ObjectsAndBackgroundsDataset(PATH_COWS, PATH_BACKGROUNDS, WEIGHT_CSV,
                                           pattern_objects=COW_IMAGE_FILE_ENDSWITH,
                                           pattern_backgrounds=None,
                                           transform=None,
                                           shrinkages_per_object=1)
    max_train = dataset._get_max_object_length()
    PATH_COWS = '/home/mroos/Data/DairyTech/labelme/scaled_segmented_train_val_sets/val/dummy'
    dataset = ObjectsAndBackgroundsDataset(PATH_COWS, PATH_BACKGROUNDS, WEIGHT_CSV,
                                           pattern_objects=COW_IMAGE_FILE_ENDSWITH,
                                           pattern_backgrounds=None,
                                           transform=None,
                                           shrinkages_per_object=1)
    max_val = dataset._get_max_object_length()
    max_obj_length = max(max_train, max_val)
    print(f'Maximum train cow image pixel length is: {max_train}')
    print(f'Maximum val cow image pixel length is:   {max_val}')
    print(f'Maximum cow image pixel length is:       {max_obj_length}')


    # Now demo actual usage...
    PATH_COWS = '/home/mroos/Data/DairyTech/labelme/scaled_segmented_train_val_sets/train/dummy'

    jitter = 0.2  # a number between 0 (no jitter) and 1 (maximum jitter)
    shrinkage = 0.5  # shrinkage is between this value and 1.0, uniformly distributed
    # shrinkage = [0.5, 0.666, 0.834, 1.0]  # strinkage is exactly these values, for each image
    if hasattr(shrinkage, '__iter__'):
        shrinkages_per_object = len(shrinkage)
        shuffle = False
    else:
        shrinkages_per_object = 1
        shuffle = True

    transform = transforms.Compose([IndependentColorJitter(brightness=jitter, contrast=jitter, saturation=jitter, hue=jitter/2),
                                    InsertObjectInBackground(224, max_obj_length, shrinkage, random_flip=True, channels_first=False, normalize=False)])

    dataset = ObjectsAndBackgroundsDataset(PATH_COWS, PATH_BACKGROUNDS, WEIGHT_CSV,
                                           pattern_objects=COW_IMAGE_FILE_ENDSWITH,
                                           pattern_backgrounds=None,
                                           transform=transform,
                                           shrinkages_per_object=shrinkages_per_object)

    # # plt.figure(1)
    # for i in range(100):
    #     sample = dataset[i]
    #     plt.clf()
    #     # plt.subplot(1,2,1)
    #     # plt.imshow(sample['image_object'])
    #     # plt.subplot(1,2,2)
    #     # plt.imshow(sample['image_background'])
    #     plt.imshow(sample['image'])
    #     plt.waitforbuttonpress()
    #     # pdb.set_trace()

    batch_size = 4
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=shuffle, num_workers=8)

    t = time.time()
    for i_batch, batch in enumerate(dataloader):
        print(i_batch)
        images, weights = batch
        plt.clf()
        for i in range(batch_size):
            plt.subplot(2, 2, i+1)
            plt.imshow(images[i])
            plt.title(f'Weight: {weights[i]:0.2f}')
            plt.axis('off')
        # plt.waitforbuttonpress()
        pdb.set_trace()
    print(f'{time.time()-t} seconds.')

