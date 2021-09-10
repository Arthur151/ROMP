# Batch Videos

Process a directory of video files into ready to run sh/bat automation files ready for further processing. Pass the ```--run_conversion``` argument to start processing the files in the same command. Otherwise only the automation files will be saved out.

# Outputs

```%animation_folders% ``` & ``` %animation_%```  represent all the matching video files found in ```--input``` . A folder will be created per video file for pipeline needs.

```
--output
	\%animation_folders%
		\convert_fbx.sh
		\proccess_mocap.sh
		\%animation_.yml%
		\%animation_results.fbx%
		\%animation_results.npz%
		\%animation_results.mp4%
	\runALL_process_convertfbx.sh
	\runALL_process_mocap.sh


```

# Automation Files
Depending on OS, the automation files will either be .bat (Windows) or .sh (Unix) files.

**Please do not move your ```ROMP/src``` folder or else these automation files will not work.** 

| name                         | description                                                  |
| ---------------------------- | ------------------------------------------------------------ |
| runALL_process_mocap.sh      | Run all the proccess_mocap.shper animation that were created. Basically batch processing the folder. Relies on your ROMP/src folder not changing paths. |
| runALL_process_convertfbx.sh | Run all the convert_fbx.sh per animation that were created. Basically batch processing the folder to create the fbx mocap. Needs .npz file per animation. |
| convert_fbx.sh               | Runs the convert_fbx.py script for the animation. Needs .npz file per animation. |
| proccess_mocap.sh            | Processes the video to an .npz file and .mp4 file then runs the  convert_fbx.py script for the animation. |



# Example
Run this example to process the videos in a directory with the extension ```.mp4``` to an output folder. Each folder will contain the process ```.fbx``` mocap file, along with the result video. 

Please run this example in the ```ROMP/src``` directory.

### Unix
```shell
python lib/utils/batch_videos.py --input=/home/user/Animations/mocap/cleaned --output=/home/user/Animations/mocap/cleaned/processed --extension mp4 --run_conversion --yaml_template=configs/video-batch.yml
```

### Windows
```sh
python lib/utils/batch_videos.py --input=M:/Animations/mocap/cleaned --output=M:/Animations/mocap/cleaned/processed --extension mp4 --windows --run_conversion --yaml_template=configs/video-batch.yml
```


# Arguments

| arg              | description                                                  | default                          |
| ---------------- | ------------------------------------------------------------ | -------------------------------- |
| --input          | Input directory containing videos to process                 | ../demo/videos/                  |
| --output         | Output directory where the newly process files will be placed | ../demo/videos/sample_video2.fbx |
| --extension      | The extension of the video files in side of the input folder. This will help the automation. | mp4                              |
| --yaml_template  | Template Yaml file to use for all the newly created .yml files. Set your settings in the template. | ROMP/src/configs/video.yml       |
| --romp_src       | The ROMP /src directory.                                     | ROMP/src                         |
| --windows        | Use this if you are running on a Windows machine.            | False                            |
| --run_conversion | Use this if you want to run the process the same time you run this script | False                            |

# Class

There's also a python ```class BatchVideos``` which you can import as a module. This is what the example script runs.

```python
batchVideos = BatchVideos(input_directory, output_directory,extension, template_yaml, test_script, windows_mode, run_conversion)
```

