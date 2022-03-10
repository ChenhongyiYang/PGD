zsh tools/dist_train.sh work_configs/det_crowdhuman/ddod_r101.py.py 8
zsh tools/dist_train.sh work_configs/pgd_ddod_crowdhuman_r101_r50.py 8
zsh tools/dist_test.sh work_configs/pgd_ddod_crowdhuman_r101_r50.py work_dirs/pgd_ddod_crowdhuman_r101_r50/epoch_12.pth 8 --out work_dirs/results.pkl
python crowd_code/eval_crowd_metric.py


