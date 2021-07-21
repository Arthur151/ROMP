
# -*- coding: utf-8 -*-

# VLT Media  LLC & Justin Jaro (VLT) is
# This script holds the same license as the program it is handling. Please contact the licensee holder through the Gihub Repo.
#
# Copyright©2021 VLT Media  LLC & Justin Jaro (VLT)
#  All rights reserved.
#
# Contact: info@vltmedia.com
#
# Author: Justin Jaro
#
# Batch convert a directory of videos into ready to process directories containing bat/sh files to automate mocap & fbx conversions. 
# User can also convert everything by passing --run_conversion

import glob
import os
import yaml
import pathlib
import argparse
import subprocess

default_yaml_template = os.path.abspath(os.path.join(os.path.dirname(__file__),'../../configs/video.yml'))
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'../../'))

class BatchVideos:
    
    def __init__(self, input_directory, output_directory,extension, template_yaml, test_script, windows_mode, run_conversion):
        self.paths = []
        self.automation_files = []
        self.automation_convertfbx_files = []
        self.run_conversion = run_conversion
        
        # Handle windows mode / bat or sh extensions
        self.end_extension = "sh"
        self.windows_mode = windows_mode
        if windows_mode == True:
            self.end_extension = "bat"
            
        self.current_yml = ""
        self.input_directory = input_directory
        self.output_directory = output_directory
        self.output_combined_bat = os.path.join(output_directory, "runALL_process_mocap." + self.end_extension)
        self.output_combined_fbx_bat = os.path.join(output_directory, "runALL_process_convertfbx." + self.end_extension)
        self.test_script = test_script
        self.extension = extension
        self.template_yaml = template_yaml
        
        # Start running the process
        self.run_process()

    def run_process(self):
        self.find_matching_files()
        self.read_template_yaml()
        self.write_vids_yaml()
        self.create_combined_bat_file()
        self.run_conversion_process()

    def find_matching_files(self):
        # Get Files in Folder
        for name in glob.glob(os.path.join(self.input_directory, "*" +self.extension)):
            self.paths.append(os.path.abspath(name))

    def read_template_yaml(self):
        f = open(self.template_yaml, "r")
        self.yaml_data = yaml.load(f, Loader=yaml.FullLoader)
        f.close()
    
    def create_bat_file(self, bat_path):
        
        bat_pathdir = os.path.dirname(bat_path)
        convert_fbx_bat_pathdir = os.path.join(os.path.dirname(bat_path) , "convert_fbx."  + self.end_extension)
        # create the processes
        move_to_script_directory = "cd "+self.test_script
        process_fbx = "\npython lib/utils/convert_fbx.py --input "+self.current_npz+" --output "+self.current_fbx_output
        process_yml = "\npython "+os.path.join(self.test_script,"core/test.py")+" --gpu=0 --configs_yml=" + self.current_yml
        if self.windows_mode == True:
            process_yml = "\npython "+os.path.join(self.test_script,"core/test_win.py")+" --gpu=0 --configs_yml=" + self.current_yml
            
        return_to_original = "\ncd "+bat_pathdir
        
        # combine and write out the proccess to a bat file
        out_regular_processing = move_to_script_directory + process_yml + process_fbx+ return_to_original
        f = open(bat_path, "w")
        f.write(out_regular_processing)
        f.close()
        
        # combine and write out the proccess to a bat file for fbx processing
        out_convert_fbx = move_to_script_directory +  process_fbx+ return_to_original
        f = open(convert_fbx_bat_pathdir, "w")
        f.write(out_convert_fbx)
        f.close()
        
        self.automation_files.append(bat_path)
        self.automation_convertfbx_files.append(convert_fbx_bat_pathdir)
        
            
    def create_combined_bat_file(self):
        
        if self.windows_mode == True:
            # Write regular process all bat
            f = open(self.output_combined_bat, "w")
            f.write("call "+"\ncall ".join(self.automation_files))
            f.close()
            
            # Write convertfbx process all bat
            f = open(self.output_combined_fbx_bat, "w")
            f.write("call "+"\ncall ".join(self.automation_convertfbx_files))
            f.close()
        else:
            # Write regular process all bat
            f = open(self.output_combined_bat, "w")
            f.write("\n".join(self.automation_files))
            f.close()
            
            # Write convertfbx process all bat
            f = open(self.output_combined_fbx_bat, "w")
            f.write("\n".join(self.automation_convertfbx_files))
            f.close()
            
            # Set Unix execute. Sets to 777, change if your security needs it
            os.chmod(self.output_combined_fbx_bat, 0o0777)
            os.chmod(self.output_combined_bat, 0o0777)
        

    def write_vids_yaml(self):
        for vid in self.paths:
            basepathh =  os.path.splitext(os.path.basename(vid))[0]
            output_folder = os.path.join(self.output_directory, basepathh).replace("\\", "/")
            self.current_yml = os.path.join(output_folder, basepathh+".yml").replace("\\", "/")
            self.current_npz = os.path.join(output_folder, basepathh+"_results.npz").replace("\\", "/")
            self.current_fbx_output = os.path.join(output_folder, basepathh+"_results.fbx").replace("\\", "/")
            output_bat = os.path.join(output_folder, "proccess_mocap." + self.end_extension).replace("\\", "/")
            # output_bat = os.path.join(output_folder, basepathh+".bat").replace("\\", "/")
            pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)
            self.yaml_data['ARGS']['input_video_path'] = vid.replace("\\", "/")
            self.yaml_data['ARGS']['output_dir'] = output_folder.replace("\\", "/")
            # # write yaml data to a file.
            yaml_file = open(self.current_yml,"w")
            # \n is placed to indicate EOL (End of Line)
            data = yaml.dump(self.yaml_data, yaml_file)
            yaml_file.close() #to change file access modes
            
            self.create_bat_file(output_bat)
            
    def run_conversion_process(self):
        if self.run_conversion:
            if self.windows_mode == True:
                subprocess.call([self.output_combined_bat])
            else:
                subprocess.call(["sh",self.output_combined_bat])



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process a directory of video files to ready to run sh/bat files ready for further processing.')
    parser.add_argument('--input', type=str, default='../demo/videos/',
                        help='Input directory containing videos to process')
    parser.add_argument('--output',  type=str, default='../demo/videos/sample_video2.fbx',
                        help='Output directory where the newly process files will be placed')
    parser.add_argument('--extension', type=str, default="mp4",
                        help='The extension of the video files in side of the input folder. This will help the automation.')
    parser.add_argument('--yaml_template', type=str, default=default_yaml_template,
                        help='Template Yaml file to use for all the newly created .yml files. Set your settings in the template.')
    parser.add_argument('--romp_src', type=str, default=root_dir,
                        help='The ROMP /src directory.')

    parser.add_argument('--windows',action='store_true',help = 'Use this if you are running on a Windows machine.')
    parser.add_argument('--run_conversion',action='store_true',help = 'Use this if you want to run the process the same time you run this script')
    
    args = parser.parse_args()
    
    
    vidconfig = BatchVideos(args.input, args.output ,args.extension, args.yaml_template , args.romp_src, args.windows, args.run_conversion)


