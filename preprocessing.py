import os
import time
import csv


def main(directory="/images"):
    """Delete the images that are broken links, so aren't useful
    directory: directory containing the images to delete"""
    dir_path = os.path.dirname(os.path.realpath(__file__))
    threshold = 5000
    print(dir_path)
    deleted_files = []
    start = time.time()
    for fn in os.listdir(dir_path + directory):
        full_path = dir_path + directory + "/" + fn
        if os.stat(full_path).st_size < threshold:
            os.remove(full_path)
            deleted_files.append(fn)
    end = time.time()
    print("Time to delete files:", round(end - start), "seconds")
    print("Number of deleted files:", len(deleted_files))
    with open("deleted_files.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows([deleted_files])


if __name__ == "__main__":
    main("/gcs_all")

