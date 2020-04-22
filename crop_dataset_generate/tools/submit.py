import json
from collections import defaultdict
import os

def main(subfix):
    base_dir = '/root/mmdetection/'
    test_1 = json.load(open('/root/ai_city/data/instances_{}.json'.format(subfix)))
    id2im_name = {i['id']: i['file_name'] for i in test_1['images']}

    print("loading bboxes...")
    res_bboxes_a = json.load(open(os.path.join(base_dir, 'reid_{}_htc.bbox.json'.format(subfix))))

    im_name2bboxes = defaultdict(list)
    cnt_filtered_a = 0
    for bbox in res_bboxes_a:
        im_name = id2im_name[bbox['image_id']]
        category_id = bbox['category_id']
        ####
        if bbox['score'] < 0.06: continue
        ###
        # if category_id in [3, 8]:
        im_name2bboxes[im_name].append([bbox['score'], category_id] + bbox['bbox'])
    cnt_filtered_b = 0

    print(len(im_name2bboxes), 'len(im_name2bboxes)')

    images, annotations = [], []
    image_id = 1
    img_counter = 0
    bbox_counter = 0
    for im_name, bboxes in im_name2bboxes.items():
        img_counter += 1
        images.append({
            "file_name": im_name,
            "id": image_id
        })
        for bbox in bboxes:
            category_id = bbox[1]
            bbox_counter += 1
            annotations.append(
                {"image_id": image_id, "bbox": bbox[2:], "category_id": category_id, "score": bbox[0]},
            )

        image_id += 1
    print("{} cnt_filtered_a".format(cnt_filtered_a))

    print(img_counter, 'img_counter')
    print("{} bboxes".format(bbox_counter))
    predictions = {"images": images, "annotations": annotations}

    save_path = os.path.join('/root/ai_city/data/AIC20_track2/AIC20_ReID/', 'reid_{}_result_0403.json'.format(subfix))
    print("save predictions to {}".format(save_path))
    with open(save_path, 'w') as f:
        json.dump(predictions, f)

if __name__ == '__main__':
    main('train')
    main('query')
    main('test')