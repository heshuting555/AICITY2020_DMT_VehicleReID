import os
from collections import defaultdict
import cv2
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm

def analyze_results(base_dir, res_file):

    imgs_dir_query = os.path.join(base_dir, 'image_query')
    imgs_dir_test = os.path.join(base_dir, 'image_test')

    file_track2 = open(res_file)
    label_images = defaultdict(list)
    for index, line in enumerate(file_track2.readlines()):
        curLine = line.strip().split(" ")
        label_images[index] = curLine

    row_id = 0
    index = 1
    col = []
    for label, images in tqdm(label_images.items()):

        img = cv2.imread(os.path.join(imgs_dir_query, str(int(label) + 1).zfill(6) + '.jpg'))
        img = cv2.resize(img, (100, 100))
        img = cv2.putText(img, str(int(label) + 1), (10, 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
        row = img
        for j in range(10):
            if j < len(images):

                img = cv2.imread(os.path.join(imgs_dir_test, images[j].zfill(6) + '.jpg'))

                img = cv2.resize(img, (100, 100))

            else:
                img = np.zeros((100, 100, 3), dtype=np.uint8)

            row = np.concatenate((row, img), axis=1)

        col.extend(row)

        row_id += 1
        if row_id % 100 == 0:
            col = np.array(col)
            cv2.imwrite(os.path.join(base_dir, 'test_{}.jpg'.format(index)), col)
            index += 1
            col = []
            row_id = 0
    col = np.array(col)
    cv2.imwrite(os.path.join(base_dir, 'test_{}.jpg'.format(index)), col)
    print('over')

if __name__ == '__main__':

    parser = ArgumentParser(description='vis txt result Tool')
    parser.add_argument('--base_dir', help='dir to the datasets images')
    parser.add_argument('--result', help='result file (txt format) path')

    args = parser.parse_args()
    analyze_results(args.base_dir, args.result)
    #analyze_results('../AIC20_track2_reid/AIC20_track2/AIC20_ReID/',
    #                os.path.join('../AIC20_track2_reid/AIC20_track2/AIC20_ReID/track2.txt'))

