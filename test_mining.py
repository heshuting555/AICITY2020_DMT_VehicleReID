import os
from torch.backends import cudnn
import numpy as np
import argparse
from datasets import make_dataloader
from model import make_model
from processor import do_inference,do_inference_query_mining
from utils.logger import setup_logger
from config import cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--thresh",  default=0.49, type=float)
    args = parser.parse_args()



    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("reid_baseline", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    train_loader, val_loader, num_query, num_classes = make_dataloader(cfg)
    model = make_model(cfg, num_class=num_classes)
    model.load_param(cfg.TEST.WEIGHT)

    distmat = do_inference_query_mining(cfg,
                 model,
                 val_loader,
                 num_query)
    #distmat = np.load(os.path.join(cfg.OUTPUT_DIR, cfg.TEST.DIST_MAT))
    print('The shape of distmat is: {}'.format(distmat.shape))
    print(distmat, 'distmat')

    thresh = args.thresh
    print('using thresh :{}'.format(thresh))
    num = 0
    query_index = []
    while num <= 333:
        if num == 0:
            query_index.append(0)
            num += 1
        max_sum = 0
        index_sum = []
        for index in range(len(distmat)):
            all_sum = 0
            flag = True
            if index not in query_index:
                for index_q in query_index:
                    all_sum += distmat[index][index_q]
                    if distmat[index][index_q] < thresh:
                        flag = False
                if all_sum > max_sum and flag:
                    max_sum = all_sum
                    index_sum.append(index)
        if index_sum == []:
            break
        else:
            index = index_sum.pop()
            query_index.append(index)
            num += 1
            #print(num,'num')

    np.save(os.path.join(cfg.OUTPUT_DIR, 'query_index_{}.npy'.format(num)) , query_index)
    print('writing result to {}'.format(os.path.join(cfg.OUTPUT_DIR, 'query_index_{}.npy'.format(num))))
    print('over')






