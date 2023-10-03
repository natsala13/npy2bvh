"""Mp4 to Bvh.

Usage:
  mp2bvh --file FILE

Options:
  -h --help       Show this screen.
  --file FILE  File to translate."""
import os

import numpy as np
from docopt import docopt

from Motion import BVH
from Motion.InverseKinematics import animation_from_positions


SMPL_JOINT_NAMES = [
    'Pelvis', # 0
    'L_Hip', # 1
    'R_Hip', # 2
    'Spine1', # 3
    'L_Knee', # 4
    'R_Knee', # 5
    'Spine2', # 6
    'L_Ankle', # 7
    'R_Ankle', # 8
    'Spine3', # 9
    'L_Foot', # 10
    'R_Foot', # 11
    'Neck', # 12
    'L_Collar', # 13
    'R_Collar', # 14
    'Head', # 15
    'L_Shoulder', # 16
    'R_Shoulder', # 17
    'L_Elbow', # 18
    'R_Elbow', # 19
    'L_Wrist', # 20
    'R_Wrist', # 21
    # 'L_Hand', # 22
    # 'R_Hand', # 23
    ]
PARENTS = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21])
HUMAN_ML_PARENTS = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19])


def load_file(filepath: str):
    _, file_extention = os.path.splitext(filepath)
    assert file_extention == '.npy', f'{file_extention} file was given, please use .npy files'

    positions = np.load(filepath, allow_pickle=True).item()
    return positions


def smpl2bvh(filepath: str):
    positions = load_file(filepath)
    motion = positions['motion']
    motion = motion.transpose(0, 3, 1, 2) # samples x joints x coord x frames ==> samples x frames x joints x coord
    
    bvh_path = filepath[:-4] + '_anim{}.bvh'
    
    for i, p in enumerate(motion):
        print(f'starting anim no. {i}, saving to {bvh_path.format(i)}')
        anim, sorted_order, _ = animation_from_positions(p, HUMAN_ML_PARENTS)
        BVH.save(bvh_path.format(i), anim, names=np.array(SMPL_JOINT_NAMES)[sorted_order])


def main():
    arguments = docopt(__doc__, version='Naval Fate 2.0')
    file = arguments['--file']
    smpl2bvh(file)

if __name__ == '__main__':
   main()
