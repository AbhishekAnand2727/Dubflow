
import shutil
import os

source_dirs = [
    r"c:\Users\anand\Documents\projects\Dubflow\Backend\Videos\uploads",
    r"c:\Users\anand\Documents\projects\Dubflow\Backend\Videos\test"
]
target_dir = r"c:\Users\anand\Documents\projects\Dubflow\Backend\Uploads"

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

for source_dir in source_dirs:
    if os.path.exists(source_dir):
        for filename in os.listdir(source_dir):
            if filename.endswith(".mp4"):
                src = os.path.join(source_dir, filename)
                dst = os.path.join(target_dir, filename)
                try:
                    if not os.path.exists(dst):
                        shutil.move(src, dst)
                        print(f"Moved {filename} to Uploads")
                    else:
                        print(f"{filename} already exists in Uploads")
                except Exception as e:
                    print(f"Error moving {filename}: {e}")
