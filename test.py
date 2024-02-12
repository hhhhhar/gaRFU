import os
import os.path as osp
import shutil

source_path = 'E:\\unity_program\\My project 1\\Assets\\SimpleMeshs'
target_folder = 'E:\\unity_program\\My project 1\\midPts'
for i in range(1, 6476):
    prefix = str(i).zfill(5)
    midIdx_path = osp.join(source_path, prefix, 'CollarMidIndex')
    if not osp.exists(osp.join(target_folder, prefix)):
        os.makedirs(osp.join(target_folder, prefix))
        target_path = osp.join(target_folder, prefix, 'CollarMidIndex')
        shutil.copy(midIdx_path, target_path)