import os
import glob
import cv2
from tqdm import tqdm

COMPRESSION_LEVEL = 3
def main():
    image_folder = "/media/my_ftp/BasesDeDatos_Torax_RX_CT/COVID19_CT/processed/object_detection/yolo_annotations/images"
    output_folder = "/media/my_ftp/BasesDeDatos_Torax_RX_CT/COVID19_CT/processed/object_detection/yolo_annotations/images_compressed_{}".format(COMPRESSION_LEVEL)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        os.makedirs(os.path.join(output_folder, "train"))
        os.makedirs(os.path.join(output_folder, "val"))

    for subfolder in ["train", "val"]:
        for filename in tqdm(glob.glob(os.path.join(image_folder, subfolder, "*.png"))):
            img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
            cv2.imwrite(os.path.join(output_folder, subfolder, os.path.basename(filename)), img, [cv2.IMWRITE_PNG_COMPRESSION, COMPRESSION_LEVEL])
    # check the number of images in both folders are the same
    assert len(glob.glob(os.path.join(image_folder, "train", "*.png"))) == len(glob.glob(os.path.join(output_folder, "train", "*.png")))
    assert len(glob.glob(os.path.join(image_folder, "val", "*.png"))) == len(glob.glob(os.path.join(output_folder, "val", "*.png")))

if __name__ == "__main__":
    main()
