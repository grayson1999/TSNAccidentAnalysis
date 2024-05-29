from mmaction.apis import inference_recognizer, init_recognizer
from mmengine import Config

# 설정 파일을 선택하고 인식기를 초기화합니다.
config = './best_model_0529/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py'
config = Config.fromfile(config)

# 로드할 체크포인트 파일을 설정합니다.
checkpoint = './best_model_0529/best_model_0529.pth'

# 인식기를 초기화합니다.
model = init_recognizer(config, checkpoint, device='cuda:0')

# 인식기를 사용하여 추론을 수행합니다.
from operator import itemgetter

test_count = 0
test5_count = 0
total_count = 0
with open("../data/custom_test_mp4.txt", 'r', encoding='utf-8') as file:
    lines = file.readlines()
    total_count = len(lines)

    for line in lines:
        video_name, video_label = line.split()

        # 예측할 비디오 파일 경로
        video = '../data/test/'+video_name
        # 라벨 파일 경로
        label = './files/index_map.txt'

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

        # 정답 lable
        # print("정답 :"+video_label)
        
        ##상위 5개 
        for result in results:
            # print(f'{result[0]}:',result[1])
            if int(video_label) == int(result[0]):
                test5_count += 1
        #상위 1개
        if int(results[0][0]) == int(video_label):
            test_count += 1
        
        print("top1: {}|top5: {}".format(test_count,test5_count))
print("top_1: {}|{} - {}%".format(test_count,total_count,test_count/total_count*100))
print("top_5: {}|{} - {}%".format(test5_count,total_count,test5_count/total_count*100))