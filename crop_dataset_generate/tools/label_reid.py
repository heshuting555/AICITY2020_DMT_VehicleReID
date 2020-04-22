import json
import os
from tqdm import tqdm
from collections import defaultdict
import cv2

def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
# Change working directory to DET dataset path
def main(subfix):
    base_dir = '/root/ai_city/data/AIC20_track2/AIC20_ReID/'
    save_dir = os.path.join(base_dir, 'image_{}_crop_0403'.format(subfix))
    check_dir(save_dir)
    image_path = os.listdir(os.path.join(base_dir, 'image_{}'.format(subfix)))

    file_path = os.path.join(base_dir, 'reid_{}_result_0403.json'.format(subfix))
    content = json.load(open(file_path, "r"))

    id2im_name_1 = {i['file_name']: i['id'] for i in content['images']}

    imlist_bbox = defaultdict(list)

    for i in range(len(content['annotations'])):
        imlist_bbox[content['annotations'][i]['image_id']].append(content['annotations'][i]['bbox'])

    annotations = []
    for imf in tqdm(image_path):
        img = cv2.imread(os.path.join(base_dir, 'image_{}'.format(subfix), imf))
        index_box = imlist_bbox[id2im_name_1[imf]]
        h, w = img.shape[:2]
        max_area = 0
        max_count = []
        for count, box in enumerate(index_box):

            area = box[2] * box[3]
            if area > max_area:
                max_area = area
                max_count.append(count)
        index = max_count.pop()

        x1 = max(int(index_box[index][0]) - 2, 0)
        y1 = max(int(index_box[index][1]) - 2, 0)
        x2 = min(int(index_box[index][0] + index_box[index][2]) + 2, w)
        y2 = min(int(index_box[index][1] + index_box[index][3]) + 2, h)

        annotations.append(
            {"file_name": imf, "bbox": [x1, y1, x2, y2]})

        crop_img = img[y1: y2, x1:x2]
        target_path = os.path.join(save_dir, imf)
        cv2.imwrite(target_path, crop_img)
    save_path = os.path.join(base_dir, 'reid_{}_bbox_result_0403.json'.format(subfix))
    print("save predictions to {}".format(save_path))
    with open(save_path, 'w') as f:
        json.dump(annotations, f)


if __name__ == '__main__':
    main('train')
    main('query')
    main('test')