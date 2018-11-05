import os
import time


def main(directory="/images"):
    """Delete the images that are broken links, so aren't useful
    directory: directory containing the images to delete"""
    dir_path = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.dirname(dir_path)
    threshold = 5000
    start = time.time()
    num_images = 0
    for fn in os.listdir(parent_dir + directory):
        num_images += 1
        full_path = parent_dir + directory + "/" + fn
        if os.stat(full_path).st_size < threshold:
            os.remove(full_path)
            print("deleted file", fn)
            print("Total files", num_images)
            curr_time = time.time()
            print("Time run so far", round(curr_time - start, "\n"))
    end = time.time()
    print("Time to delete files:", round(end - start), "seconds")
    print("Number of images", num_images)


if __name__ == "__main__":
    main("/gcs_all")

