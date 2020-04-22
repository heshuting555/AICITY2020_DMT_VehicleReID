# encoding: utf-8

import glob
import xml.dom.minidom as XD
import os.path as osp
from .bases import BaseImageDataset


class AIC(BaseImageDataset):
    """
    VR

    Dataset statistics:

    """
    dataset_dir = 'AIC20_track2/AIC20_ReID'

    # dataset_dir_test = './data/VeRi'

    def __init__(self, root='../data', verbose=True, **kwargs):
        super(AIC, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_track_path = osp.join(self.dataset_dir, 'train_track_id.txt')
        self.test_track_path = osp.join(self.dataset_dir, 'test_track_id.txt')

        self.train_dir = osp.join(self.dataset_dir, 'image_train')
        self.query_dir = osp.join(self.dataset_dir, 'image_query')
        self.gallery_dir = osp.join(self.dataset_dir, 'image_test')

        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir_test(self.query_dir, relabel=False)
        gallery = self._process_dir_test(self.gallery_dir, relabel=False, query=False)


        if verbose:
            print("=> AIC loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False, if_track=False):
        xml_dir = osp.join(self.dataset_dir, 'train_label.xml')
        info = XD.parse(xml_dir).documentElement.getElementsByTagName('Item')

        pid_container = set()
        for element in range(len(info)):
            pid = int(info[element].getAttribute('vehicleID'))
            if pid == -1: continue  # junk images are just ignored
            # if pid > 200: continue
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        _, _, frame2trackID = self._process_track(path=self.train_track_path) #### Revised

        for element in range(len(info)):
            pid, camid = map(int, [info[element].getAttribute('vehicleID'), info[element].getAttribute('cameraID')[1:]])
            image_name = str(info[element].getAttribute('imageName'))
            if pid == -1: continue  # junk images are just ignored
            # if pid > 200: continue
            if relabel: pid = pid2label[pid]
            trackid = frame2trackID[int(image_name[:-4])]
            dataset.append((osp.join(dir_path, image_name), pid, camid,trackid))
        return dataset

    def _process_dir_test(self, dir_path, relabel=False, query=True):
        img_paths = sorted(glob.glob(osp.join(dir_path, '*.jpg')))

        dataset = []
        _, _, frame2trackID = self._process_track(path=self.test_track_path)
        for img_path in img_paths:
            camid = 1
            pid = 2
            if query:
                dataset.append((img_path, pid, camid, -1))
            else:
                trackid = frame2trackID[int(img_path[-10:-4])]
                dataset.append((img_path, pid, camid, trackid))
        #print(len(dataset), 'len(dataset)')
        return dataset


    def _process_track(self,path): #### Revised

        file = open(path)
        tracklet = dict()
        frame2trackID = dict()
        nums = []
        for track_id, line in enumerate(file.readlines()):
            curLine = line.strip().split(" ")
            nums.append(len(curLine))
            curLine = list(map(eval, curLine))
            tracklet[track_id] =  curLine
            for frame in curLine:
                frame2trackID[frame] = track_id
        return tracklet, nums, frame2trackID