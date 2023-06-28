from datasets.image_base import Image_base, test_image_dataset
from datasets.image_base_relative import Image_base_relative, test_image_relative_dataset
from datasets.video_base_relative import Video_base_relative, test_video_relative_dataset

Base_Classes = {'image': Image_base, 'image_relative': Image_base_relative, 'video_relative': Video_base_relative}

Test_Funcs = {'image': test_image_dataset, 'image_relative': test_image_relative_dataset, 'video_relative': test_video_relative_dataset}