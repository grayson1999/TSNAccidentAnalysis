import optuna
from mmengine import Config
import os.path as osp
import mmengine
from mmengine.runner import Runner

# Optuna의 objective 함수를 정의합니다.
def objective(trial):
    # 하이퍼파라미터 최적화: 학습률 및 배치 크기
    lr = trial.suggest_loguniform('lr', 1e-6, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16, 32])

    # 설정 파일 로드 및 수정
    cfg = Config.fromfile('./best_model_0527/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py')
    cfg.optim_wrapper.optimizer.lr = lr
    cfg.train_dataloader.batch_size = batch_size
    
    # 데이터 경로 설정
    cfg.data_root = '/mmaction2/data/train/'
    cfg.data_root_val = '/mmaction2/data/val/'
    cfg.ann_file_train = '/mmaction2/data/custom_train_mp4.txt'
    cfg.ann_file_val = '/mmaction2/data/custom_val_mp4.txt'
    cfg.test_dataloader.dataset.ann_file = '/mmaction2/data/custom_val_mp4.txt'
    cfg.test_dataloader.dataset.data_prefix.video = '/mmaction2/data/val/'
    cfg.train_dataloader.dataset.ann_file = '/mmaction2/data/custom_train_mp4.txt'
    cfg.train_dataloader.dataset.data_prefix.video = '/mmaction2/data/train/'
    cfg.val_dataloader.dataset.ann_file = '/mmaction2/data/custom_val_mp4.txt'
    cfg.val_dataloader.dataset.data_prefix.video = '/mmaction2/data/val/'

    # 모델의 클래스 수를 수정합니다.
    cfg.model.cls_head.num_classes = 434

    # 사전 학습된 TSN 모델을 사용합니다.
    cfg.load_from = './best_model_0527/best_acc_top1_epoch_15.pth'

    # 파일과 로그를 저장할 작업 디렉토리를 설정합니다.
    cfg.work_dir = './best_model'

    # 하이퍼파라미터를 설정합니다.
    cfg.optim_wrapper.optimizer.lr = lr
    cfg.train_dataloader.batch_size = batch_size
    cfg.val_dataloader.batch_size = batch_size

    # 원래 학습률(LR)은 8-GPU 학습을 위해 설정되어 있습니다.
    # 우리는 1개의 GPU만 사용하기 때문에 8로 나눕니다.
    cfg.optim_wrapper.optimizer.lr = cfg.optim_wrapper.optimizer.lr / 8
    cfg.train_cfg.max_epochs = 20

    # 데이터 로더의 작업자 수를 설정합니다.
    cfg.train_dataloader.num_workers = 1
    cfg.val_dataloader.num_workers = 0
    cfg.test_dataloader.num_workers = 0

    # 작업 디렉토리를 생성합니다.
    mmengine.mkdir_or_exist(osp.abspath(cfg.work_dir))

    # 설정에서 러너를 빌드합니다.
    runner = Runner.from_cfg(cfg)

    # 학습을 시작합니다.
    runner.train()

    # 검증 정확도를 반환하여 Optuna가 이를 최적화합니다.
    # 검증 결과는 러너의 로그 파일이나 모델 결과에서 추출할 수 있습니다.
    validation_results = runner.validate()
    top1_acc = validation_results.get('top1_acc', 0)
    top5_acc = validation_results.get('top5_acc', 0)
    mean_acc = validation_results.get('mean_acc', 0)
    memory = validation_results.get('memory', 0)
    print(top1_acc, top5_acc, mean_acc, memory)

    # 최적화 목표를 top1_acc로 설정합니다.
    return top1_acc

# Optuna 스터디를 생성하고 최적화를 시작합니다.
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# 최적의 하이퍼파라미터 출력
print(f"Best hyperparameters: {study.best_params}")
print(f"Best validation accuracy (top1_acc): {study.best_value}")


    # # 설정 파일을 불러옵니다.
    # cfg = Config.fromfile('/AccidentFaultAI/model/TSN/best_model_0527/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py')

    # # 데이터셋 타입과 경로를 수정합니다.
    # cfg.data_root = '/AccidentFaultAI/datasets/data/video_datasets/download_datas/train/'
    # cfg.data_root_val = '/AccidentFaultAI/datasets/data/video_datasets/download_datas/val/'
    # cfg.ann_file_train = '/AccidentFaultAI/datasets/data/video_datasets/download_datas/custom_train_mp4.txt'
    # cfg.ann_file_val = '/AccidentFaultAI/datasets/data/video_datasets/download_datas/custom_val_mp4.txt'
    
    # # 테스트 데이터 로더의 데이터셋 주석 파일 및 데이터 경로를 수정합니다.
    # cfg.test_dataloader.dataset.ann_file = '/AccidentFaultAI/datasets/data/video_datasets/download_datas/custom_val_mp4.txt'
    # cfg.test_dataloader.dataset.data_prefix.video = '/AccidentFaultAI/datasets/data/video_datasets/download_datas/val/'

    # # 훈련 데이터 로더의 데이터셋 주석 파일 및 데이터 경로를 수정합니다.
    # cfg.train_dataloader.dataset.ann_file = '/AccidentFaultAI/datasets/data/video_datasets/download_datas/custom_train_mp4.txt'
    # cfg.train_dataloader.dataset.data_prefix.video = '/AccidentFaultAI/datasets/data/video_datasets/download_datas/train/'

    # # 검증 데이터 로더의 데이터셋 주석 파일 및 데이터 경로를 수정합니다.
    # cfg.val_dataloader.dataset.ann_file = '/AccidentFaultAI/datasets/data/video_datasets/download_datas/custom_val_mp4.txt'
    # cfg.val_dataloader.dataset.data_prefix.video = '/AccidentFaultAI/datasets/data/video_datasets/download_datas/val/'
