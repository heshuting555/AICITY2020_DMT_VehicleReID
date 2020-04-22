import os
import json
import time
from PIL import Image

def main(subfix):
    start = time.time()
    categories = []
    data = {}
    data['info'] = {"description": "COCO 2017 Dataset", "url": "http://cocodataset.org", "version": "1.0",
                    "year": 2019, "contributor": "COCO Consortium", "date_created": "2019/08/18"}
    data['licenses'] = [{"url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
                         "id": 1, "name": "Attribution-NonCommercial-ShareAlike License"}]

    images = []
    annotations = []
    outfile_1 = open('/root/ai_city/data/instances_{}.json'.format(subfix), 'w')

    base_dir = '/root/ai_city/data/AIC20_track2/AIC20_ReID'
    imfiles = os.listdir(os.path.join(base_dir,'image_{}'.format(subfix)))
    index = 0

    for imf in imfiles:
        # if imf in ex_path:
        #  continue
        im = Image.open(os.path.join(base_dir,'image_{}'.format(subfix), imf))
        item = {}
        item['license'] = 1
        item['file_name'] = imf
        item['width'], item['height'] = im.size

        item['id'] = index
        images.append(item)
        index += 1
    for v in range(1, 81):
        # print(v)
        category = {}
        category['id'] = v
        category['name'] = str(v)
        category['supercategory'] = 'defect_name'
        categories.append(category)
    data['images'] = images

    data['categories'] = categories

    json.dump(data, outfile_1, indent=4)
    print('------------------------------------one-------------------------------------------')
    print('generate  json time: %fs' % (time.time() - start))


if __name__ == '__main__':
    main('train')
    main('query')
    main('test')

