import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
import random
import os
from PIL import Image

## Just for development
import matplotlib.pyplot as plt
plt.ion()


PATH_COWS = '/home/mroos/Data/DairyTech/labelme/scaled_segmented'
COW_IMAGE_FILE_ENDSWITH = 'cow_side_seg.tiff'

PATH_BACKGROUNDS = '/Data/DairyTech/Flickr_fields'


class ObjectsAndBackgroundsDataset(Dataset):
    '''
    Images of segemented objects paired with background images
    into which they can be inserted.
    '''

    def __init__(self, path_objects, path_backgrounds, pattern_objects=None, pattern_backgrounds=None, transform=None):
        random.seed(0)

        # Any transforms will be applied to object and background images independently
        self.transform = transform

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
        Because we want to effectively randomly match object images with background
        images we make the length very large, effectively infinite. And we match images
        using modulos of the index.
        '''
        return len(np.iinfo(np.int32).max)

    def __getitem__(self, idx):
        '''
        Because we want to effectively randomly match object images with background
        images we make the length very large, effectively infinite. And we match images
        using modulos of the index.
        '''
        if torch.is_tensor(idx):
            idx = idx.tolist()

        idx_obj = idx % self.n_objects
        idx_bg = idx % self.n_backgrounds

        # Get object image
        im_obj = Image.open(self.filenames_objects[idx_obj])

        # Create object segmentation mask from image stored as numpy array
        np_filename = self.filenames_objects[idx_obj][:-4]  + 'npy'
        np_obj = np.load(np_filename)
        mask = np.logical_not(np.all(np_obj==-1.0, axis=2))

        # Get background image
        im_bg = Image.open(self.filenames_backgrounds[idx_bg])

        sample = {'image_object':im_obj, 'mask_object':mask, 'image_background':im_bg}

        if self.transform:
            sample = self.transform(sample)

        return sample


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

    def __init__(self, output_size, object_shrinkage):
        '''
        output_size: int indicating output image size (3, output_size, output_size)
                     where 3 is the number of channels.
        object_shrinkage: factor indicating how small the object may be resized
                          to, relative to the output_size. E.g., if output_size
                          is 1000 and object_shrinkage is 0.5 then the largest
                          dimension of the object will be shrunk to between
                          0.5*1000==500 and 1000.
        '''
        self.output_size = output_size
        self.object_shrinkage = object_shrinkage
        self.crop_transform = transforms.RandomResizedCrop(224, scale=(0.5, 1.0), ratio=(0.75, 1.33))

    def __call__(self, sample):
        bg = self.crop_transform(sample['image_background'])
        area_original = np.sum(sample['mask_object'])
        
        # Determine the new size of the object image
        h_orig = sample['image_object'].height
        w_orig = sample['image_object'].width
        f_shrink = np.random.uniform(self.object_shrinkage, 1.0)
        f_resize = 224/max(h_orig, w_orig) * f_shrink
        h_new = round(h_orig * f_resize)
        w_new = round(w_orig * f_resize)

        # Resize the object
        sample['image_object'] = sample['image_object'].resize((w_new, h_new), resample=Image.BILINEAR)

        # Also resize the mask (so we can count pixels from that).
        # Convert mask to uints of 0 and 255 in order to use PIL to resize.
        mask = Image.fromarray((sample['mask_object']*255).astype(np.uint8))
        mask = mask.resize((w_new, h_new), resample=Image.BILINEAR)
        mask = np.array(mask) > 0
        area_resized = np.sum(mask)


        ## TODO
        # Insert masked object into random location in background


        return {'image':bg}



if __name__ == "__main__":

    jitter = 0.0  # a number between 0 (no jitter) and 1 (maximum jitter)
    # transform = IndependentColorJitter(brightness=jitter, contrast=jitter, saturation=jitter, hue=jitter/2)
    transform = transforms.Compose([IndependentColorJitter(brightness=jitter, contrast=jitter, saturation=jitter, hue=jitter/2),
                                    InsertObjectInBackground(224, 0.5)])

    dataset = ObjectsAndBackgroundsDataset(PATH_COWS, PATH_BACKGROUNDS,
                                           pattern_objects=COW_IMAGE_FILE_ENDSWITH,
                                           pattern_backgrounds=None,
                                           transform=transform)

    # plt.figure(1)
    for i in range(100):
        sample = dataset[i]
        plt.clf()
        plt.subplot(1,2,1)
        plt.imshow(sample['image_object'])
        plt.subplot(1,2,2)
        plt.imshow(sample['image_background'])
        import pdb
        pdb.set_trace()


