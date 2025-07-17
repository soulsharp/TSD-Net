import torch.utils.data as data
import os
from PIL import Image, ImageTransform
from torchvision.transforms import v2 
import torchvision.transforms as transforms
import torch

class DatasetFromFolder(data.Dataset):
    def __init__(self, base_dir, classification_only_flag = False, transform=None, mask_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.base_dir = base_dir
        self.classification_only_flag = classification_only_flag  
        self.samples = self.load_data() 
        self.transform = transform
        self.mask_transform = mask_transform  
        
        # By default, the image and mask are resized to (224, 224)
        self.default_image_transform = v2.Compose([
        v2.Resize((224, 224), interpolation=v2.InterpolationMode.BILINEAR),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
        ])

        self.default_mask_transform = v2.Compose([
        v2.Resize((224, 224), interpolation=v2.InterpolationMode.NEAREST),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
        ])

    def load_data(self):
        samples = []
        
        # Folder paths
        good_images_path = os.path.join(self.base_dir, "good")
        bad_images_path = os.path.join(self.base_dir, "bad")

        # Puts "good" images in the dataset   
        if os.path.exists(good_images_path):
            img_list = os.listdir(good_images_path)

            assert len(img_list) > 0, "Image list should not be empty"
            img_list = sorted(img_list)

            for file_path in img_list:
                full_path = os.path.join(good_images_path, file_path)
                sample_obj = {"img_path" : full_path, 
                            "label" : 0, 
                            "mask_path" : None}
                samples.append(sample_obj)

        else:
            raise FileNotFoundError(f"The path {good_images_path} does not exist")
        
        # Puts "bad" images into the dataset
        if os.path.exists(bad_images_path):
            img_list = os.listdir(bad_images_path)
            img_list = sorted(img_list) 
            assert len(img_list) > 0, "Image list should not be empty"

            # When only classification is needed
            if self.classification_only_flag:
                for file_path in img_list:
                    full_path = os.path.join(bad_images_path, file_path)
                    sample_obj = {"img_path" : full_path, 
                                "label" : 1, 
                                "mask_path" : None}
                    samples.append(sample_obj)
            
            # Includes masks corresponding to the image in the "mask" field
            else:
                masks_path = os.path.join(self.base_dir, "masks")
                mask_list = sorted(os.listdir(masks_path))
                assert len(mask_list) == len(img_list), "Masks and image counts are not equal"

                for idx in range(len(mask_list)):
                    sample_obj = {"img_path" : os.path.join(bad_images_path, img_list[idx]), 
                                   "label" : 1, 
                                   "mask_path" : os.path.join(masks_path, mask_list[idx])}
                    samples.append(sample_obj)
            
        else:
            raise FileNotFoundError(f"The path {bad_images_path} does not exist")
        
        return samples
    
    def __len__(self):
        return len(self.samples)

    def shape_image(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample["img_path"]).convert('RGB')
        return img.size
   
    def __getitem__(self, idx):
        sample = self.samples[idx]
        assert sample["img_path"] is not None, "Image path should not be none"
        image = Image.open(sample["img_path"]).convert('RGB')
        label = sample["label"]

        image = torch.as_tensor(self.default_image_transform(image))
        
        mask = None
        if sample["mask_path"] is not None:
            mask = Image.open(sample["mask_path"]).convert('RGB')
            if self.mask_transform is not None:
                mask = torch.as_tensor(self.mask_transform(mask))
            else:
                mask = torch.as_tensor(self.default_mask_transform(mask))
        
        else:
            mask = torch.zeros((1, 224, 224), dtype=torch.float32)

        return {
            "image": image,
            "label": label,
            "mask": mask
        }