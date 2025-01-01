from typing import List

import torch
import numpy as np
from einops import rearrange

from mp2bvh.Motion import BVH
from mp2bvh.Motion.Quaternions import Quaternions
from mp2bvh.Motion.Animation import positions_global
from mp2bvh.Motion.AnimationStructure import get_kinematic_chain
from mp2bvh.Motion.InverseKinematics import animation_from_positions
from mp2bvh.utils.plot_script import plot_3d_motion
from mp2bvh.utils.motion_transformation import recover_from_ric


PARENTS = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8,
                     9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21])
HUMAN_ML_PARENTS = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5,
                              6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19])
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

class DynamicMotion:
    def __init__(self, positions: np.ndarray, parents: np.ndarray=PARENTS):
        self.positions = positions
        self.parents = parents

    @classmethod
    def init_from_bvh(cls, filepath: str):
        '''Load motion from .bvh file'''
        assert filepath.endswith('.bvh'), f'{filepath} is not a .bvh file'
        animation, _, _ = BVH.load(filepath)
        positions = positions_global(animation)
        return cls(positions, animation.parents)
    
    @classmethod
    def init_from_npy(cls, filepath: str, parents: np.ndarray=HUMAN_ML_PARENTS):
        '''Load motion from .npy file'''
        assert filepath.endswith('.npy'), f'{filepath} is not a .npy file'
        
        positions = np.load(filepath, allow_pickle=True)
        try:
            positions = positions.item()
            motion = positions['motion']
            motion = motion.transpose(0, 3, 1, 2) # B, J, f, T ==> B, T, J, f
        except Exception as e:
            motion = positions[np.newaxis, :]
        
        return cls(motion, parents)
    
    @classmethod
    def init_from_humanml(cls, filepath: str):
        '''Load motion from .npy file'''
        assert filepath.endswith('.npy'), f'{filepath} is not a .npy file'
        
        motion = np.load(filepath, allow_pickle=True)
        
        positions = recover_from_ric(torch.from_numpy(motion), 22)
        
        return cls(positions.numpy(), HUMAN_ML_PARENTS)
    
    @classmethod
    def init_from_xia(cls, filepath: str):
        '''Load motion from .npy file'''
        assert filepath.endswith('.npy'), f'{filepath} is not a .npy file'
        motion = np.load(filepath, allow_pickle=True)
        
        features = motion[:, :-4]  # Remove feet contact...
        feature_matrix = rearrange(features, 'time (joints features) -> time joints features',
                                   joints=21, features=11)

        body = feature_matrix[:, :, :7]
        traj = feature_matrix[:, :, 7:]
                              
        positions = cls.restore_animation_from_xia(body[:, :, :3], traj)
        
        return cls(positions, HUMAN_ML_PARENTS)

    @property
    def kinematic_tree(self) -> List[List[int]]:
        return get_kinematic_chain(self.parents)

    def to_bvh(self, filepath: str) -> None:
        '''Save the dynamic motion to a .bvh file.'''
        animation, sorted_order, _ = animation_from_positions(self.positions[0], self.parents)
        # save_path = filepath[:-4] + '_anim{}.bvh'
        BVH.save(filepath, animation, names=np.array(SMPL_JOINT_NAMES)[sorted_order])
        print(f'Saved bvh file to {filepath}')

    def to_mp4(self, filepath: str, title: str='') -> None:
        '''Save the dynamic motion to a .mp4 file.'''
        plot_3d_motion(filepath, self.kinematic_tree,
                        self.positions, title=title, fps=20)
        print(f'Saved mp4 file to {filepath}')
    
    @staticmethod
    def restore_animation_from_xia(pos, traj, start=None, end=None):
        """
        :param pos: (F, J, 3)
        :param traj: (F, J, 4)
        :param start: start frame index
        :param end: end frame index
        :return: positions
        """
        if start is None:
            start = 0
        if end is None:
            end = len(pos)

        Rx = traj[start:end, 0, -4]
        Ry = traj[start:end, 0, -3]
        Rz = traj[start:end, 0, -2]
        Rr = traj[start:end, 0, -1]

        rotation = Quaternions.id(1)
        translation = np.array([[0, 0, 0]])

        for fi in range(len(pos)):
            pos[fi, :, :] = rotation * pos[fi]
            pos[fi] = pos[fi] + translation[0]  # NOTE: xyz-translation
            rotation = Quaternions.from_angle_axis(-Rr[fi], np.array([0, 1, 0])) * rotation
            translation = translation + rotation * np.array([Rx[fi], Ry[fi], Rz[fi]])
        global_positions = pos

        return global_positions