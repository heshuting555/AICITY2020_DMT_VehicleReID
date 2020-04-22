echo Generate crop dataset

python tools/test-generate.py

python tools/test.py configs/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e_test.py checkpoints/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e_20190408-0e50669c.pth --out reid_test_htc.json

python tools/test.py configs/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e_query.py checkpoints/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e_20190408-0e50669c.pth --out reid_query_htc.json

python tools/test.py configs/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e_train.py checkpoints/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e_20190408-0e50669c.pth --out reid_train_htc.json

python tools/submit.py

python tools/label_reid.py
