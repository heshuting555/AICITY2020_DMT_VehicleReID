# encoding: utf-8

import glob
import xml.dom.minidom as XD
import os.path as osp
from .bases import BaseImageDataset
import numpy as np
import os


class AIC_Q(BaseImageDataset):
    """
    VR

    Dataset statistics:

    """
    dataset_dir_sim = '../data/AIC20_track2/AIC20_ReID_Simulation'
    dataset_dir = 'AIC20_track2/AIC20_ReID'
    # You should run 'python test_mining.py' first to get selected query id.
    load_path = os.path.join('/data/model/0409_2/', 'query_index_189.npy')

    def __init__(self, root='../data', verbose=True, **kwargs):
        super(AIC_Q, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_track_path = osp.join(self.dataset_dir, 'train_track_id.txt')
        self.test_track_path = osp.join(self.dataset_dir, 'test_track_id.txt')

        self.train_dir = osp.join(self.dataset_dir, 'image_train')
        self.query_dir = osp.join(self.dataset_dir, 'image_query')
        self.gallery_dir = osp.join(self.dataset_dir, 'image_test')
        self.plus_num_id = 100
        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir_query(self.query_dir)
        gallery = self._process_dir_test(self.gallery_dir)

        self.train = train
        self.query = query
        self.gallery = gallery

        if verbose:
            print("=> VR loaded")
            self.print_dataset_statistics(train, query, gallery)
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

    def _process_dir(self, dir_path, relabel=False):
        xml_dir = osp.join(self.dataset_dir, 'train_label.xml')
        info = XD.parse(xml_dir).documentElement.getElementsByTagName('Item')

        pid_container = set()
        for element in range(len(info)):
            pid = int(info[element].getAttribute('vehicleID'))
            if pid == -1: continue  # junk images are just ignored
            # if pid > 200: continue
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        num_class = len(pid_container)
        dataset = []
        _, _, frame2trackID = self._process_track(path=self.train_track_path) #### Revised
        for element in range(len(info)):
            pid, camid = map(int, [info[element].getAttribute('vehicleID'), info[element].getAttribute('cameraID')[1:]])
            image_name = str(info[element].getAttribute('imageName'))
            if pid == -1: continue  # junk images are just ignored
            # if pid > 200: continue
            if relabel: pid = pid2label[pid]
            trackid = frame2trackID[int(image_name[:-4])]
            dataset.append((osp.join(dir_path, image_name), pid, camid, trackid))

        xml_dir = osp.join(self.dataset_dir_sim, 'train_label.xml')
        info = XD.parse(xml_dir).documentElement.getElementsByTagName('Item')

        pid_container = set()
        for element in range(len(info)):
            pid = int(info[element].getAttribute('vehicleID'))
            if pid == -1: continue  # junk images are just ignored
            if pid > self.plus_num_id: continue
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        for element in range(len(info)):
            pid, camid = map(int, [info[element].getAttribute('vehicleID'), info[element].getAttribute('cameraID')[1:]])
            image_name = str(info[element].getAttribute('imageName'))
            if pid == -1: continue  # junk images are just ignored
            # if pid > 200: continue
            if pid > self.plus_num_id: continue
            if relabel: pid = pid2label[pid]
            dataset.append((osp.join(self.dataset_dir_sim,'image_train', image_name), pid + num_class, camid, 1))
        return dataset

    def _process_dir_test(self, dir_path):
        img_paths = sorted(glob.glob(osp.join(dir_path, '*.jpg')))

        dataset = []
        _, _, frame2trackID = self._process_track(path=self.test_track_path)
        for img_path in img_paths:
            camid = 1
            pid = 2

            trackid = frame2trackID[int(img_path[-10:-4])]
            dataset.append((img_path, pid, camid, trackid))
        #print(len(dataset), 'len(dataset)')
        return dataset

    def _process_dir_query(self, dir_path):
        img_paths = sorted(glob.glob(osp.join(dir_path, '*.jpg')))
        dataset = []
        _, _, frame2trackID = self._process_track(path=self.test_track_path)
        for img_path in img_paths:
            camid = 1
            pid = 2
            trackid = -1
            dataset.append((img_path, pid, camid, trackid))

        query_index = np.load(self.load_path)
        print('loading query_index result from:{}'.format(self.load_path))
        query_container=[]
        for index in query_index:
            query_container.append(dataset[index])
        print('length of query_container is :{}'.format(len(query_container)))
        return query_container

    def _process_track(self, path):  #### Revised

            file = open(path)
            tracklet = dict()
            frame2trackID = dict()
            nums = []
            for track_id, line in enumerate(file.readlines()):
                curLine = line.strip().split(" ")
                nums.append(len(curLine))
                curLine = list(map(eval, curLine))
                tracklet[track_id] = curLine
                for frame in curLine:
                    frame2trackID[frame] = track_id
            return tracklet, nums, frame2trackID
