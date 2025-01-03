import os
import cv2 as cv
import glob
import numpy as np

images = []

img_folder_paths = sorted([f"extrinsics/{i}" for i in os.listdir("extrinsics") if not i.startswith('.')])
print(img_folder_paths)

for img_folder_path in img_folder_paths:
    image_names = sorted(glob.glob(f'./{img_folder_path}/*.jpg'))
    print(image_names)
    imgs = []
    for fname in image_names:
        img = cv.imread(fname)
        # cv.imshow(f'{fname}',img)
        imgs.append(img)
        # key = cv.waitKey(0) & 0xFF
        # if key == ord('q'):
        #     print("Exiting...")
        #     cv.destroyAllWindows()
        #     quit()
    images.append(imgs)

images = np.array(images)
print(images.shape)
images = images.swapaxes(0,1)
print(images.shape)