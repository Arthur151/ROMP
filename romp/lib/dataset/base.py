from dataset.image_base import Image_base, test_image_dataset
from dataset.image_base_relative import Image_base_relative, test_image_relative_dataset
from config import args

Base_Classes = {'image': Image_base, 'image_relative': Image_base_relative}

Test_Funcs = {'image': test_image_dataset, 'image_relative': test_image_relative_dataset}