from mmengine import Config
import os.path as osp
import mmengine
from mmengine.runner import Runner
from mmengine import Config
from mmengine.runner import set_random_seed

# 설정 파일을 불러옵니다.
cfg = Config.fromfile('../configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py')

# 데이터셋 타입과 경로를 수정합니다.


cfg.data_root = '/mmaction2/data/train/'
cfg.data_root_val = '/mmaction2/data/val/'
cfg.ann_file_train = '/mmaction2/data/custom_train_mp4.txt'
cfg.ann_file_val = '/mmaction2/data/custom_val_mp4.txt'

# 테스트 데이터 로더의 데이터셋 주석 파일 및 데이터 경로를 수정합니다.
cfg.test_dataloader.dataset.ann_file = '/mmaction2/data/custom_val_mp4.txt'
cfg.test_dataloader.dataset.data_prefix.video = '/mmaction2/data/val/'

# 훈련 데이터 로더의 데이터셋 주석 파일 및 데이터 경로를 수정합니다.
cfg.train_dataloader.dataset.ann_file = '/mmaction2/data/custom_train_mp4.txt'
cfg.train_dataloader.dataset.data_prefix.video = '/mmaction2/data/train/'

# 검증 데이터 로더의 데이터셋 주석 파일 및 데이터 경로를 수정합니다.
cfg.val_dataloader.dataset.ann_file = '/mmaction2/data/custom_val_mp4.txt'
cfg.val_dataloader.dataset.data_prefix.video = '/mmaction2/data/val/'

# 모델의 클래스 수를 수정합니다.
cfg.model.cls_head.num_classes = 434

# 사전 학습된 TSN 모델을 사용합니다.
##이어서 학습
cfg.load_from = '/mmaction2/grayson/work_space/best_acc_top1_epoch_8.pth'

# 파일과 로그를 저장할 작업 디렉토리를 설정합니다.
cfg.work_dir = './work_space_2'

# 원래 학습률(LR)은 8-GPU 학습을 위해 설정되어 있습니다.
# 우리는 1개의 GPU만 사용하기 때문에 8로 나눕니다.
cfg.train_dataloader.batch_size = cfg.train_dataloader.batch_size // 16
cfg.val_dataloader.batch_size = cfg.val_dataloader.batch_size // 16
cfg.optim_wrapper.optimizer.lr = cfg.optim_wrapper.optimizer.lr / 8 / 16
cfg.train_cfg.max_epochs = 20

# 데이터 로더의 작업자 수를 설정합니다.
cfg.train_dataloader.num_workers = 1
cfg.val_dataloader.num_workers = 0
cfg.test_dataloader.num_workers = 0

# 학습을 위한 로거를 초기화하고 최종 설정을 출력합니다.
print(f'Config:\n{cfg.pretty_text}')

# 작업 디렉토리를 생성합니다.
mmengine.mkdir_or_exist(osp.abspath(cfg.work_dir))

# 설정에서 러너를 빌드합니다.
runner = Runner.from_cfg(cfg)

# 학습을 시작합니다.
runner.train()

# 테스트를 실행합니다.
runner.test()