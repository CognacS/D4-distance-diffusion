import os
import shutil

def create_folder(folder: str):

    # check existance
    if not os.path.isdir(folder):
        # create directory
        os.makedirs(folder)



def remove_all_files_in_dir(folder: str):

    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))