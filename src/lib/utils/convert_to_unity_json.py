# Imports
import numpy as np
import os
import json
import sys
# noinspection PyPep8Naming
from scipy.spatial.transform import Rotation


# This class converts AMASS SMPLH .npz body animation files into Unity-readable .json files.
# See AMASSConverterExamples file for an example on how to use this class.
class AMASSDataConverter:

    # SMPLH Parameters
    JOINTS = 52
    ROTATION_VECTOR_DIMENSIONS = 3
    QUATERNION_DIMENSIONS = 4

    # Local variable initialization for static-typing (stupid python)
    gender: np.ndarray
    betas: np.ndarray
    poses: np.ndarray
    dmpls: np.ndarray
    trans: np.ndarray
    frames: int
    fps: int
    poses_as_quaternion: np.ndarray

    # Constructor for class
    def __init__(self, npz_file_path: str, show_messages=True):
        self.show_messages = show_messages
        self.npzFile = npz_file_path

        # Load npz file
        self.data = self.load_data()

        # Read data from loaded file
        self.read_data()

        # AMASS poses are exponential rotation vectors, unity needs quaternions.
        reshaped_poses = np.reshape(self.poses, [self.poses.shape[0], self.JOINTS, self.ROTATION_VECTOR_DIMENSIONS])
        self.convert_poses_to_quaternions(reshaped_poses)

        # convert data to dicts, since JSON is dictionary-based format
        self.data_as_dict = {
            "gender": self.gender,
            "trans": self.trans,
            "poses": self.poses_as_quaternion,
            "betas": self.betas,
            "dmpls": self.dmpls,
            "fps": self.fps,
        }

    # Converts poses from exponential rotation vectors to quaternions
    def convert_poses_to_quaternions(self, reshaped_poses):
        poses_as_quaternion = np.empty([self.frames, self.JOINTS, self.QUATERNION_DIMENSIONS])
        for frameIndex in range(0, self.frames):
            for poseIndex in range(0, self.JOINTS):
                rotation_vector_raw = reshaped_poses[frameIndex][poseIndex]
                rotation = Rotation.from_rotvec(rotation_vector_raw)
                quaternion = rotation.as_quat()
                poses_as_quaternion[frameIndex][poseIndex] = quaternion
        self.poses_as_quaternion = poses_as_quaternion

    # This sets up the converter from numpy arrays to json. Will spit error if not numpy.
    @staticmethod
    def default_encoding(obj):
        if type(obj).__module__ == np.__name__:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj.item()
        raise TypeError('Unknown type:', type(obj))

    # Reads data from loaded npz file and creates internal objects
    def read_data(self):
        self.gender = 'male'
        #self.data['gender'].astype(str)
        self.betas = self.data['betas']
        self.poses = self.data['poses']
        self.dmpls = self.data['dmpls']
        self.trans = self.data['trans']
        self.frames = self.poses.shape[0]
        self.fps = round(int(np.rint(self.data['mocap_framerate'])))
        if self.show_messages:
            print(f"Created converter for file: {self.npzFile}")
            print(f'\tgender: {self.gender}')
            print(f'\tbetas: {self.betas.shape}')
            print(f'\tposes: {self.poses.shape}')
            print(f'\tdmpls: {self.dmpls.shape}')
            print(f'\ttrans: {self.trans.shape}')
            print(f'\tframes detected: {self.frames}')
            print(f'\tfps: {self.fps}')

    # Loads npz file into data structure
    def load_data(self):
        # noinspection PyBroadException
        try:
            data = np.load(self.npzFile)
            return data
        except Exception:
            print(f'Could not read {self.npzFile}! Skipping...')

    # Finishes conversion and saves dicts into JSON format
    def write_to_json(self, json_path: str):
        if self.show_messages:
            print(f"\nWriting to json... {json_path}")
        filename, file_extension = os.path.splitext(json_path)

        if file_extension != '.json':
            print(f"Incorrect extension specified ({file_extension}). Needs to be .json.")
            return

        dumped = json.dumps(self.data_as_dict, default=self.default_encoding, indent=4)
        with open(json_path, 'w') as f:
            f.write(dumped)
        if self.show_messages:
            print('\n  *** DONE CONVERSION ***\n')


def main():
    if len(sys.argv) != 3:
        print("Not the right number of arguments. First should be source npz file path, second should be destination "
              "json path.")
        return
    converter = AMASSDataConverter(sys.argv[1])
    converter.write_to_json(sys.argv[2])


if __name__ == "__main__":
    main()