# %% I need to copy the following folders to /dev/shm/ozavala/data
import os
from os.path import join

in_folders = ['/unity/f1/ozavala/DATA/GOFFISH/AVISO/GoM',
              '/unity/f1/ozavala/DATA/GOFFISH/EddyDetection/PreprocContours_ALL_1993_2022']

output_folders = ['/dev/shm/ozavala/data','/conda/ozavala/data']


for in_folder in in_folders:
    for root_out_folder in output_folders:
        last_folder = in_folder.split('/')[-1]
        out_folder = join(root_out_folder, last_folder)
        if os.path.exists(out_folder):
            # Do you want to remove folder? 
            answer = input(f"Do you want to remove folder {out_folder}? (y/n): ")
            if answer == 'y':
                cmd = f"rm -rf {out_folder}"
                print(f"Removing folder {cmd}")
                os.system(cmd)
                print("Done!")
                os.makedirs(out_folder)
        else:
            os.makedirs(out_folder)

        # Copy force folder recursively
        cmd = f"cp -r {in_folder} {root_out_folder}"
        print(cmd)
        os.system(cmd)

        # Mv entire folder one level up
        # cmd = f"mv {out_folder}/*.* {'/'.join(out_folder.split('/')[:-1])}"
        # print(cmd)
        # cmd = f"rmdir {'/'.join(out_folder.split('/')[:-1])}"
        # print(cmd)

print("Done!")
# %%
