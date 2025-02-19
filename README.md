해당 프로그램은 [Video-Swin-Transformer](https://github.com/SwinTransformer/Video-Swin-Transformer)를 참고하여 제작했습니다.

## TSN (Temporal Segment Networks)

<aside>
💡 비디오를 여러 세그먼트로 나누어 각 세그먼트의 특징을 추출한 후 통합하여 행동 인식을 수행하는 딥러닝 모델
이 방식은 비디오의 시간적, 공간적 정보를 효과적으로 활용하여 높은 인식 성능을 제공
</aside>
<br>

자세한 내용은 
[GitHub - SwinTransformer/Video-Swin-Transformer: This is an official implementation for "Video Swin Transformers](https://github.com/SwinTransformer/Video-Swin-Transformer)
을 참조하세요


### 목차
1. [목적](#목적)
2. [모델](#모델)
3. [환경 설정](#환경-설정) 
4. [DATA SET](#data-set) 
5. [Model 학습 방법](#model-학습-방법) 
6. [tester(테스터기)](#tester테스터기) 
7. [recognizor(추론기)](#recognizor추론기) 


### 목적
<aside>
434가지의 사고 유형을 인식하여 비디오를 학습하고, 주어진 비디오에서 가장 유사한 사고 유형을 탐지하는 것
</aside>

### 모델
| 모델 이름             | 정확도(top1) | 정확도(top5) | 평균 정확도(mean1) | 로스   | 메모리 |
|----------------------|--------------|--------------|--------------------|--------|--------|
<<<<<<< HEAD
| best_model_0522 | 0.2061   | 0.3876       | 0.29685          | 3.6529 | 353 MB |
| best_model_0527 | 0.2304   | 0.4683       |  0.34935         | 3.4279 | 353 MB |
=======
| best_model_0522 | 0.2061   | 0.3876       | 0.29685         | 3.6529 | 353 MB |
| best_model_0527 | 0.2304   | 0.4683       | 0.34935         | 3.4279 | 353 MB |
| best_model_0529 | 0.2056   | 0.4503       | 0.0364          | 3.4289 | 353 MB |
| best_model_0531 | 0.1857   | 0.4206       | 0.0333          | 0.3735 | 320 MB |


### 환경 설정

mmaction2 설치 가이드

[Installation — MMAction2 1.2.0 documentation](https://mmaction2.readthedocs.io/en/latest/get_started/installation.html)

torch+torchvision 설치 방법

```bash
##torch+torchvision
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
##mmcv 설치
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html

##추가 모듈 설치
pip install opencv-python
pip install timm
pip install scipy
pip install einops

##오류 대응
pip install numpy==1.19.0
```

Docker 이미지
- 버전 수정
    ```bash
    ARG PYTORCH="1.6.0"
    ARG CUDA="10.1"
    ARG CUDNN="7"
    ```
- **Important:** Make sure you've installed the [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).
- docker 빌드
    
    ```bash
    # build an image with PyTorch 1.6.0, CUDA 10.1, CUDNN 7.
    docker build -f ./docker/Dockerfile --rm -t mmaction2 .
    
    # docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/mmaction2/data mmaction2
    docker run --gpus all --shm-size=8g -it -v G:/video_datasets/download_datas:/mmaction2/data mmaction2
    
    pip install mmcv==2.1.0
    pip install -r requirements/build.txt
    python setup.py develop
    
    apt-get update
    apt-get install wget
    ```
    

### DATA SET

- download
    
    [AI-Hub](https://www.aihub.or.kr/devsport/apishell/list.do?currMenu=403&topMenu=100)
    
    ```bash
    export AIHUB_ID=''
    export AIHUB_PW=''
    aihubshell -mode d -datasetkey 597 -filekey 509338
    ```
        

- 데이터 셋 구성 방법
    
    [https://github.com/SwinTransformer/Video-Swin-Transformer/blob/master/docs/tutorials/3_new_dataset.md](https://github.com/SwinTransformer/Video-Swin-Transformer/blob/master/docs/tutorials/3_new_dataset.md)
    
    - download 폴더 구성
    
    ```markdown
    ### download 시                      ### anotation 변환
    Root                                 Root
    ├── origin                           ├── train
    │   └── subfolder                    │    └── *.mp4
    │       └── *.mp4                    ├── val
    │                                    │    └── *.mp4 
    └── label                            ├── test
        └── subfolder                    │    └── *.mp4 
    	    └── *.json                     ├── custom_train_mp4.txt
    	                                   ├── custom_val_mp4.txt
    	                                   └── custom_test_mp4.txt
    ```
    
    - video_annotion 변환 방법
        - 변환 방법
            
            ```bash
            python {Download folder}/convert_video_annotation.py
            ```
            
        - train :  val : test = 70 : 15 : 15 비율로 작성 함
        - videodataset 방식의 annotation 진행
    - annotation 형식
        
        ```
        bb_1_210121_two-wheeled-vehicle_236_21840.mp4 206
        bb_1_211031_two-wheeled-vehicle_241_21549.mp4 232
        bb_1_210125_two-wheeled-vehicle_112_003.mp4 290
        bb_1_210917_two-wheeled-vehicle_121_126.mp4 298
        ...
        ```
        

### Model 학습 방법

- tutorial
    
    [Google Colab Tutorial](https://colab.research.google.com/drive/1dLeCGfq3bQFpgtfU5WSPFlvkKsZCWsdo#scrollTo=VcjSRFELVbNk)
    
1. 사전 학습 된 TSN 가중치 다운로드(optional)
    
    ```bash
    mkdir checkpoints
    wget -c https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth \
          -O ./checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth
    ```
    
2. config 수정 및 학습
    
    ```python
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
    # cfg.load_from = './checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth'
    
    # 파일과 로그를 저장할 작업 디렉토리를 설정합니다.
    cfg.work_dir = './work_space'
    
    # 원래 학습률(LR)은 8-GPU 학습을 위해 설정되어 있습니다.
    # 우리는 1개의 GPU만 사용하기 때문에 8로 나눕니다.
    cfg.train_dataloader.batch_size = cfg.train_dataloader.batch_size // 16
    cfg.val_dataloader.batch_size = cfg.val_dataloader.batch_size // 16
    cfg.optim_wrapper.optimizer.lr = cfg.optim_wrapper.optimizer.lr / 8 / 16
    cfg.train_cfg.max_epochs = 50
    
    # 데이터 로더의 작업자 수를 설정합니다.
    cfg.train_dataloader.num_workers = 2
    cfg.val_dataloader.num_workers = 2
    cfg.test_dataloader.num_workers = 2
    
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
    ```
    

### tester(테스터기)

```python
from mmaction.apis import inference_recognizer, init_recognizer
from mmengine import Config

# 설정 파일을 선택하고 인식기를 초기화합니다.
config = './sample_work/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py'
config = Config.fromfile(config)

# 로드할 체크포인트 파일을 설정합니다.
checkpoint = './sample_work/best_acc_top1_epoch_9.pth'

# 인식기를 초기화합니다.
model = init_recognizer(config, checkpoint, device='cuda:0')

# 인식기를 사용하여 추론을 수행합니다.
from operator import itemgetter

test_count = 0
total_count = 0
with open("../data/custom_test_mp4.txt", 'r', encoding='utf-8') as file:
    lines = file.readlines()
    total_count = len(lines)

    for line in lines:
        video_name, video_label = line.split()

        # 예측할 비디오 파일 경로
        video = '../data/test/'+video_name
        # 라벨 파일 경로
        label = './index_map.txt'

        # 비디오에 대한 인식 결과를 얻습니다.
        results = inference_recognizer(model, video)

        # 예측 점수를 리스트로 변환합니다.
        pred_scores = results.pred_score.tolist()
        # 예측 점수와 인덱스를 튜플로 묶습니다.
        score_tuples = tuple(zip(range(len(pred_scores)), pred_scores))
        # 점수를 기준으로 내림차순 정렬합니다.
        score_sorted = sorted(score_tuples, key=itemgetter(1), reverse=True)
        # 상위 5개의 라벨을 선택합니다.
        top5_label = score_sorted[:5]

        # 라벨 파일을 읽어옵니다.
        labels = open(label).readlines()
        # 라벨에서 공백 문자를 제거합니다.
        labels = [x.strip() for x in labels]

        # 상위 5개 라벨과 점수를 매핑합니다.
        results = [(labels[k[0]], k[1]) for k in top5_label]

        # 상위 1개 가져오기
        print("정답 :"+video_label)
        print(f'{results[0][0]}: ', results[0][1])

        if int(results[0][0]) == int(video_label):
            test_count += 1
print("{}|{} - {}%".format(test_count,total_count,test_count/total_count*100))
```

### recognizor(추론기)

1. config
    - 학습 시 사용한 workspace에 생성되어 있는  config 파일 사용
2. checkpoint 
    - workspace에 생성 된 best 가중치 사용
3. label
    - 0~433, 총 434개의 숫자가 “\n”으로 분리된 파일로 /data 폴더에 같이 저장되어 있음

```python
from mmaction.apis import inference_recognizer, init_recognizer
from mmengine import Config

# 설정 파일을 선택하고 인식기를 초기화합니다.
config = './sample_work/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py'
config = Config.fromfile(config)

# 로드할 체크포인트 파일을 설정합니다.
checkpoint = './sample_work/best_acc_top1_epoch_8.pth'

# 인식기를 초기화합니다.
model = init_recognizer(config, checkpoint, device='cuda:0')

# 인식기를 사용하여 추론을 수행합니다.
from operator import itemgetter

# 예측할 비디오 파일 경로
video = './test2_175.mp4'
# 라벨 파일 경로
label = './index_map.txt'

# 비디오에 대한 인식 결과를 얻습니다.
results = inference_recognizer(model, video)

# 예측 점수를 리스트로 변환합니다.
pred_scores = results.pred_score.tolist()
# 예측 점수와 인덱스를 튜플로 묶습니다.
score_tuples = tuple(zip(range(len(pred_scores)), pred_scores))
# 점수를 기준으로 내림차순 정렬합니다.
score_sorted = sorted(score_tuples, key=itemgetter(1), reverse=True)
# 상위 5개의 라벨을 선택합니다.
top5_label = score_sorted[:5]

# 라벨 파일을 읽어옵니다.
labels = open(label).readlines()
# 라벨에서 공백 문자를 제거합니다.
labels = [x.strip() for x in labels]

# 상위 5개 라벨과 점수를 매핑합니다.
results = [(labels[k[0]], k[1]) for k in top5_label]

# 상위 5개 라벨과 해당 점수를 출력합니다.
print('The top-5 labels with corresponding scores are:')
for result in results:
    print(f'{result[0]}: ', result[1])
```

오류 모음

1. GPG 에러
    
    ```bash
    오류 내용:
    GPG error: https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64  InRelease: The following signatures couldn't be verified because the public key is not available: NO_PUBKEY A4B469963BF863CC
    해결 방법:
    # NVIDIA CUDA 리포지토리의 공개 키 다운로드 및 추가
    RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC 
    ```
    
    [GPG error: https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64  InRelease: The following signatures couldn't be verified because the public key is not available: NO_PUBKEY A4B469963BF863CC](https://better-tomorrow.tistory.com/entry/GPG-error-httpsdeveloperdownloadnvidiacomcomputecudareposubuntu1804x8664-InRelease-The-following-signatures-couldnt-be-verified-because-the-public-key-is-not-available-NOPUBKEY-A4B469963BF863CC)
    
2. numpy 버전 에러
    
    ```bash
    ##오류 내용
    AttributeError: module 'numpy' has no attribute 'int'.
    `np.int` was a deprecated alias for the builtin `int`. To avoid this error in existing code, use `int` by itself. Doing this will not modify any behavior and is safe. When replacing `np.int`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.
    The aliases was originally deprecated in NumPy 1.20; for more details and guidance see the original release note at:
        https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
    ```
    
    ```bash
    ##해결 방법
    pip install numpy==1.19.0
    ```

### version
| 버전       | 날짜      | 변경 내용                                |
|------------|-------------|------------------------------------------|
|ver 1.0|24.05.26|video-swin-transformer를 이용해 과실 측정 모델 제작|
|ver 1.1|24.05.26|docker file 수정|
|ver 1.2|24.05.26|test top5 섹션 추가, 모델 명 변경|
|ver 1.3|24.05.28|otuna 하이퍼파라미터 최적화 알고리즘 작성|
|ver 1.4|24.05.29|best_model_0529 모델 추가|
|ver 1.5|24.05.31|best_model_0531 모델 추가|
