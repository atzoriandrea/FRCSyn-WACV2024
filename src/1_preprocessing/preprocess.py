import gc
import multiprocessing
import os.path
import sys
import traceback
from PIL import Image
from tqdm import tqdm
import cv2
import numpy as np
import torch
from deepface import DeepFace
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def get_files_full_path(rootdir):
    import os
    paths = []
    for root, dirs, files in os.walk(rootdir):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                paths.append(os.path.join(root, file))
    return paths


def resume(imgs, cropped):
    import os
    basepath = "/".join(imgs[0][::-1].split("/")[3:])[::-1]
    relatives, images = [], []
    for crp in cropped:
        relatives.append("/".join(crp.split("/")[-3::]))
    for img in imgs:
        images.append("/".join(img.split("/")[-3::]))

    diff = list(set(images).difference(relatives))

    for i, d in enumerate(diff):
        diff[i] = os.path.join(basepath, d)
    return diff


def exclude(imgs, exclude_file):
    files = []
    with open(exclude_file, "r") as ex:
        for r in ex:
            files.append(r.strip())
    diff = list(set(imgs).difference(files))
    diff_n = len(imgs) - len(diff)
    print("Excluded %d files" % diff_n)
    return diff



def preprocess(images, resizeonly, proc_id):
    with tqdm(total=len(images), desc="Process %d" % proc_id) as pbar:
        backends = ['opencv', 'ssd', 'dlib', 'mtcnn']
        b_id = 3
        for image in images:
            try:
                outfilepath = image.split("/")
                outfilepath[-4] = "Aligned_" + outfilepath[-4]
                sub_root_folder = "/".join(outfilepath[:-1])
                outfilepath = "/".join(outfilepath)

                if not os.path.exists(outfilepath):
                    if not resizeonly:
                        cropped_image = DeepFace.detectFace(image, detector_backend=backends[b_id],
                                                            target_size=(112, 112), enforce_detection=False,
                                                            align=True)

                        if cropped_image is None:
                            img = Image.open(image)

                            # Get the dimensions of the image
                            width, height = img.size

                            # Determine the side length for the square image
                            side_length = max(width, height)

                            # Create a new square canvas with a black background
                            square_img = Image.new('RGB', (side_length, side_length), (0, 0, 0))

                            # Calculate the position to paste the original image
                            paste_x = (side_length - width) // 2
                            paste_y = (side_length - height) // 2

                            # Paste the original image onto the square canvas
                            square_img.paste(img, (paste_x, paste_y))

                            # Resize the square image to 112x112
                            square_img = square_img.resize((112, 112), Image.Resampling.LANCZOS)
                            cropped_image = np.asarray(square_img, dtype=np.uint8)

                        # det = dlib.rectangle(int(- 56), int(-56), int(56), int(56))
                        # cropped_image = detect(face_detector, backends[b_id], image)
                    else:
                        cropped_image = cv2.resize(cv2.imread(image), (36, 36))

                    if not os.path.exists(sub_root_folder):
                        os.makedirs(sub_root_folder)
                        # print("Creted folder: %s " % sub_root_folder)
                    # cv2.imwrite(outfilepath, cv2.normalize(cropped_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))
                    cv2.imwrite(outfilepath,
                                cv2.normalize(cropped_image[:, :, ::-1], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))
                    gc.collect()
                    # print("Processing time: " + str(time.time() - start_time))
            except Exception as e:
                print(e)
                traceback.print_exc()
                gc.collect()
                torch.cuda.empty_cache()
                # pass
                # print("Invalid Image: " + image)
            pbar.update(1)


if __name__ == '__main__':
    import argparse

    try:
        parser = argparse.ArgumentParser(description='Image aligner and cropper')
        parser.add_argument('--imgs', metavar='path', required=True,
                            help='path to images')
        parser.add_argument('--processes', required=True, default=6,
                            help='number of cpu processes')
        parser.add_argument('--resizeonly', required=False, default=False,
                            help='set as True if you just want to resize yoor images')
        parser.add_argument('--exclude', required=False, default=None,
                            help='list of files to exclude')
        args = parser.parse_args()
        if args.imgs == "":
            print("Invalid image path")
    except Exception as e:
        print(e)
        sys.exit(1)
    images = get_files_full_path(args.imgs)[::-1]
    outfilepath = images[0].split("/")
    outfilepath[-4] = "Aligned_" + outfilepath[-4]
    outfilepath = "/".join(outfilepath[:-3])
    done = get_files_full_path(outfilepath)
    if len(done) > 0:
        images = resume(images, done)
        print("Already done: %d images - remaining: %d images" % (len(done), len(images)))
    if args.exclude is not None:
        pre = len(images)
        images = exclude(images, args.exclude)
        print("Excluded: %d images - remaining: %d images" % (pre, len(images)))
    num_processes = int(args.processes)
    chunk_length = int(len(images) / num_processes)
    image_chunks = []
    for rank in range(num_processes):
        if rank != num_processes - 1:
            image_chunks.append(images[rank * chunk_length:(rank + 1) * chunk_length])
        else:
            image_chunks.append(images[rank * chunk_length:])
    processes = []

    if num_processes != 1:
        for i, c in enumerate(image_chunks):
            processes.append(
                multiprocessing.Process(target=preprocess, args=(c, bool(args.resizeonly), i)))
    else:
        preprocess(image_chunks[0], bool(args.resizeonly), 0)

    for t in processes:
        t.start()
    for t in processes:
        t.join()
