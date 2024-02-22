import numpy as np
# import einops
from Motion import BVH, Animation

from Motion.Animation import positions_global
from Motion.AnimationStructure import get_kinematic_chain
from Motion.InverseKinematics import animation_from_positions
from mp2bvh.plot_script import plot_3d_motion


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
    def __init__(self, positions: np.ndarray, parents: list[int]=PARENTS):
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
    def init_from_npy(cls, filepath: str, parents: list[int]=HUMAN_ML_PARENTS):
        '''Load motion from .npy file'''
        assert filepath.endswith('.npy'), f'{filepath} is not a .npy file'
        
        positions = np.load(filepath, allow_pickle=True).item()
        
        motion = positions['motion']
        motion = motion.transpose(0, 3, 1, 2) # B, J, f, T ==> B, T, J, f
        return cls(motion, parents)

    @property
    def kinematic_tree(self) -> list[list[int]]:
        return get_kinematic_chain(self.parents)

    def to_bvh(self, filepath: str) -> None:
        '''Save the dynamic motion to a .bvh file.'''
        animation, sorted_order, _ = animation_from_positions(self.positions, self.parents)
        # save_path = filepath[:-4] + '_anim{}.bvh'
        BVH.save(filepath, animation, names=np.array(SMPL_JOINT_NAMES)[sorted_order])

    def to_mp4(self, filepath: str, title: str='') -> None:
        '''Save the dynamic motion to a .mp4 file.'''
        plot_3d_motion(filepath, self.kinematic_tree,
                        self.positions, title=title, fps=20)

