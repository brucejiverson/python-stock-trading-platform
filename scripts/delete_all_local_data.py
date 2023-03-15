# deletes all the files except for .gitignore in the data folder
import os
import shutil


def delete_file(file_path):
    print(f'Deleting {file_path}')
    try:
        os.remove(file_path)
    except OSError as e:
        print("Error: %s : %s" % (file_path, e.strerror))
        

def delete_all_files_in_folder(folder_path):
    # walk over the subdirectories and files and delete
    for root, dirs, files in os.walk(folder_path):
        if root == folder_path:
            continue
        
        for file in files:
            if file == '.gitignore':
                continue
            
            delete_file(os.path.join(root, file))
            
            
if __name__ == "__main__":
    delete_all_files_in_folder('data')
    delete_all_files_in_folder('logs')    
    