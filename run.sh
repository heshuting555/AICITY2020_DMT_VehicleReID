echo AICITY online score reproduction

echo Train

python train.py --config_file='configs/baseline_aic.yml' OUTPUT_DIR "('/data/model/0402_5/')" ### here we get the model resnet101_ibn_a_80.pth in the 0402_5.

python train.py --config_file='configs/baseline_aic_finetune.yml' OUTPUT_DIR "('/data/model/0402_6/')"  ### here we get the model resnet101_ibn_a_40.pth in the 0402_6.

python test_mining.py --config_file='configs/test_identity_mining.yml' ### here we get the selected query id in the 0409_2.

python train_IM.py --config_file='configs/baseline_aic.yml' --config_file_test='configs/test_train_IM.yml' OUTPUT_DIR "('/data/model/0407_1/')"  ### here we get the model resnet101_ibn_a_80.pth in the 0407_1.

python train_IM.py --config_file='configs/baseline_aic_se.yml' --config_file_test='configs/test_train_IM.yml' OUTPUT_DIR "('/data/model/0408_6/')"  ### here we get the model se_resnet101_ibn_a_80.pth in the 0408_6.

python train.py --config_file='configs/baseline_aic.yml' INPUT.SIZE_TRAIN "([384,384])" INPUT.SIZE_TEST "([384,384])" DATALOADER.NUM_INSTANCE "(6)" SOLVER.IMS_PER_BATCH "(60)" OUTPUT_DIR "('/data/model/0409_2/')"  ### here we get the model resnet101_ibn_a_80.pth in the 0409_2.

python train.py --config_file='configs/baseline_aic.yml' DATASETS.NAMES "('aic_crop')" INPUT.SIZE_TRAIN "([384,384])" INPUT.SIZE_TEST "([384,384])" DATALOADER.NUM_INSTANCE "(6)" SOLVER.IMS_PER_BATCH "(60)" OUTPUT_DIR "('/data/model/0409_3/')"  ### here we get the model resnet101_ibn_a_80.pth in the 0409_3.
# You can cd crop_dataset_generate folder and running "bash run.sh" to get the aic_crop datasets.

echo Test
# here we will get Distmat Matrix after test.
python test.py --config_file='configs/baseline_aic.yml' TEST.RE_RANKING_TRACK "(True)" TEST.WEIGHT "('/data/model/0402_6/resnet101_ibn_a_40.pth')" OUTPUT_DIR "('/data/model/0402_6/')"

python test.py --config_file='configs/baseline_aic.yml' TEST.RE_RANKING_TRACK "(True)" TEST.WEIGHT "('/data/model/0407_1/resnet101_ibn_a_80.pth')" OUTPUT_DIR "('/data/model/0407_1/')"

python test.py --config_file='configs/baseline_aic_se.yml' TEST.RE_RANKING_TRACK "(True)" TEST.WEIGHT "('/data/model/0408_6/se_resnet101_ibn_a_80.pth')"  OUTPUT_DIR "('/data/model/0408_6/')"

python test.py --config_file='configs/baseline_aic.yml' TEST.RE_RANKING_TRACK "(True)" TEST.WEIGHT "('/data/model/0409_2/resnet101_ibn_a_80.pth')" INPUT.SIZE_TRAIN "([384,384])" INPUT.SIZE_TEST "([384,384])"  OUTPUT_DIR "('/data/model/0409_2/')"

python test.py --config_file='configs/baseline_aic.yml' TEST.RE_RANKING_TRACK "(True)" TEST.WEIGHT "('/data/model/0409_3/resnet101_ibn_a_70.pth')" DATASETS.NAMES "('aic_crop')" INPUT.SIZE_TRAIN "([384,384])" INPUT.SIZE_TEST "([384,384])" OUTPUT_DIR "('/data/model/0409_3/')"

echo Ensemble

python ensemble_dist.py