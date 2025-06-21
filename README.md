# 학교 식당 식기반납대의 컨베이어 벨트 위를 이동하는 식판 위에서 카드와 같은 분실물을 카메라를 통해 인식하기.

소프트웨어융합개론 Term-project

2019102077 김민규 2019102099 손지원 2019102136 최성준

## 1. 주제 선정 이유

식판 위에 카드를 놓고 밥을 먹은 후 식기반납대에 그대로 반납하여 카드를 분실한 경험이 있다. 식판이 설거지 공간으로 들어가기 전에 분실물을 미리 인식할 수 있다면, 직원분들이 분실물을 분류하지 않아도 되며 학생들은 물건을 분실 할 위험을 줄일 수 있다.
따라서 카메라 영상처리를 공부하여 물체인식을 해보고자 이러한 주제를 선택했고, 이를 공부하며 기계학습을 통해 물체를 구별하는 소프트웨어를 만들고 보고자 하였다.

## 2. 개발 내용

### 2. (1) 카드의 외곽선 검출하기

카드를 인식하기 위해서는 다른 물체와 카드의 차이를 알고, 그 차이를 통해 카드를 인식하게 해야 한다. 우리가 생각한 신용 카드의 특징은 4개의 꼭지점을 가지고 있는 직사각형의 형태이고, 비교적 그 경계선이 명확하며, 카드의 가로세로 비율은 황금비 ( 1: 1.618 ) 이다. 따라서 직사각형의 형태가 아닌 것들은 모두 베제시키고, 휴대폰 등은 경계가 명확하지 않으므로 제외된다.

### --이 코드는 opencv 라이브러리가 설치되어 있어야 한다.--

우선 Videocapture 함수를 이용하여 웹캠 ( 노트북의 경우 노트북 카메라 ) 을 실행시키고, while 문을 이용해 연속적으로 루프를 돌면서 프레임을 받는다. 각각의 프레임에 대해 모든 경계선들을 추출하고, 경계선 끼리의 오차가 크지 않다면 approxPolyDP 함수를 이용해 꼭짓점 수를 줄여 새로운 다각형을 만든다. 이때 카드를 분류하기 위해 변의 개수가 4인 데이터만을 추출할 것이고, 사각형 중에 카드 안의 글자 또는 네모난 칩 등을 제외하기 위해 화면 전체 비율의 일정 부분을 넘어설 때만 데이터를 추출한다. 만약 추출한 데이터가 없을 때에는 웹캠 화면 그대로 계속 보여주고, 추출한 외곽이 존재하다면 웹캠 화면 위에 외곽을 drawContours 함수를 이용해 보여준다.

### 하지만 2. (1) 의 과정까지 실행했을 때 카드의 외곽 중 일부분이 손으로 가려지거나 다른 물체로 가려졌을 때 카드로 인식하지 못한다는 한계점이 있었다.

## 2. (2) 딥러닝을 통한 객체 인식

2.(1) 의 과정까지 마친 후 해당 결과를 이용해 소프트웨어 패스티벌에 전시했었는데, 이대호 교수님께서 YOLO 알고리즘을 사용하면 다른 물체로 테두리 부분이 가려졌을 때에도 카드를 인식할 수 있다고 조언해 주셨고, YOLO 알고리즘을 사용해 딥 러닝을 하고, 해당 학습을 통해 카드를 더 정확하게 인식을 할 수 있게 했다.

### 1단계: Labeling

많은 양의 카드가 들어가 있는 사진 데이터에서 카드가 있는 부분만을 labeling 하여 컴퓨터에게 학습시키는 YOLO 알고리즘을 사용하기 위해서는 우선 많은 양의 카드가 들어가있는 사진 데이터를 다운받아야 한다. 따라서 우리는 웹 크롤링을 통해 많은 양의 사진 데이터를 불러왔다.

웹 크롤링을 통해 url에 포함되어 있는 모든 정보를 크롤링, 스크레이핑 한 후 그 정보들 중에서 카드가 포함된 이미지인 jpg, png 파일들만 부분적으로 뽑아내는 과정을 거쳤다. 이 과정을 통해 200여장의 자료를 수집하긴 했지만, 이대호 교수님이 말씀해주신 Deep Learning을 수행하기 위한 최소 10000장이상의 사진을 얻기란 너무 복잡한 과정이었다. 따라서 웹브라우저를 원격조작에 사용하는 selenium, object detecting Dataset관련 사이트인 KITTI 2D Dataset, OpenImages 등에서 정보를 수집해보려 하였으나 원하는 card 데이터를 수집하기란 어려웠다. 다양한 방법을 시도해보던 중 google chrome extention을 활용해보기로 결정했고 1500장 가량을 수집할 수 있었다.

이제 해당 웹 크롤링을 통해 받은 데이터는 jpg의 형태와 png의 형태로 이루어져 있는데, 모든 png 파일을 jpg 파일로 변환하여 획일화 시킨다. 그리고 anaconda prompt를 이용해 labelImg 을 실행한다. 

conda install pyqt=5 

pyrcc5 -o libs/resources.py resources.qrc 

python labelImg.py 

그리고 labelImg 프로그램 중 Create\RectBox 기능을 통해 한 사진에 하나 또는 하나 이상의 카드 영역을 추출한다. 그리고 카드 영역 추출이 완료된 데이터를 xml 형식의 파일로 저장한다. 이때 xml 파일 안에 있는 데이터는 추출한 카드 영역이 해당 이미지의 어느 부분에 있는지 나타내 준다. 이때 직사각형의 양 꼭지점의 x좌표와 y좌표, 전체 이미지의 크기 등을 저장하고 있다.

이때 데이터 처리의 편의를 위해 이미지의 크기는 width = 416, height = 416, depth = 3으로 고정시킨다. 

분류하고 있는 데이터의 class는 카드 하나이므로 모두 credit_card 이라는 class이다.

### 2단계: Deep learning

코드 및 데이터는 keras-yolo3-master폴더로 이메일 첨부하겠습니다.

논문을 통해 yolo알고리즘의 원리를 파악하고 오픈소스내 파일의 상관관계를 분석하는 과정을 통해 card detection에 적용할 수 있도록 재구성하였다. 

1. 기존의 영상자료만 사용할 수 있었던 코드를 리니어하여 webcam으로 real-time object detection을 하도록 변형했다.

2. labelImg한 자료만으로는 학습시키기에 역부족이라는 판단하에 coco dataset에서 학습시킨 데이터를 pretraining 과정을 통해 weight의 초기값으로 설정해주었다.

3. card_data의 labelImg한 자료들을 학습시키는 과정을 거치며 최종 card detection하도록 설계하였다.

#### keras-yolo3-master 파일구조 (상관관계)

- __pycache__
  * yolo.cpython-36.pyc
- .vscode
  * settings.json
- card_data
  * *.jpg
  * *.xml
- data_access
  * annotation.txt - labeling했던 좌표가 저장. 각 xml 파일의 필요한 정보를 뽑은 후 txt 파일로 저장. 이것을 train.py가 가져가서 처리하는 것.
  * classes.txt
- font
  * FiraMono-Medium.otf
  * SIL Open FOnt License.txt
- logs - 학습시킬 때 로그 저장 파일이며, 모델을 그때그때 저장해 주는 것.
  * 000
  * ep~~~
- model_data
  * card_model.h5 - yolo_weights.h5, yolo.h5파일로 pretrained 작업을 통해 우리가 도출해낸 파일. 학습이 잘되었는지 yolo_video.py, yolo.py파일을 기반으로 실행해보는 것.
  * card_model.png, retrained_model.png - 신경망 모델을 학습 시킬 때 연산(tensor 연산(gradient descent, 미분 등)을 계산)의 효율성을 위해서 그래프로 나타냄. iteration만 하면 되게 직관적으로 자료구조를 바꿔놓아 그래프로 visualization한 것이 png 파일들.
  * coco_classes.txt
  * tiny_yolo_anchors.txt
  * voc_classes.txt
  * yolo_anchors.txt
  * yolo_weights.h5, yolo.h5 - 학습을 원활하게 시키기 위해 pretrained 작업을 거침. dataset 중 VOC와 coco의 weight와 architecture를 먼저 학습시켜서 저장해 놓은 것이 yolo_weights.h5, yolo.h5 파일.
- yolo3
  * __pycache__
  * __init__.cpython-36.pyc
  * model.cpython-36.pyc
  * utils.cpython-36.pyc
  * __init__.py
  * model.py - 가장 중요. train.py의 기저를 둠. 학습시킬 yolo모델을 담아놓은 파일. yolo도 학습을 시킬거면 모델이 있어야함. 신경망 architecture구조를 정의해 놓은 파일
  * utils.py
- .gitignore
- convert.py
- darknet53.cfg
- kmeans.py
- LICENSE
- README.md
- train_bottleneck.py
- train.py - 학습의 중추. 이 파일을 실행시켜서 학습을 시킴. model.py의 로스를 계산하거나 학습을 진행하는데 톱니바퀴같은 역할. 이렇게 학습이 된 파일은 model_data폴더의 card_model.h5 파일에 중추가 저장이 됨. 학습시킨 모델을 재사용 가능.
- card_annotation.py - pascal VOC 형식으로 직접 labeling한 데이터를 학습 가능한 텍스트 파일명으로 바꾸는 작업.
- coco_annotation.py - coco dataset에서 labeling한 데이터를 학습 가능한 텍스트 파일명으로 바꾸는 작업. (실제 이번프로젝트에서 pretrained할 때 coco dataset을 사용함.)
- voc_annotation.py - pascal VOC dataset에서 labeling한 데이터를 학습 가능한 텍스트 파일명으로 바꾸는 작업.
- yolo_video.py - yolo.py를 vedio파일 리니어한 버전에서는 웹캠에 끌어와서 쓸 수 있게끔.(video와 웹캠에서 실시간 object detection을 가능하게 함.)
- yolo.py - training시킨 모델을 직접적으로 yolo에 활용해보는 class를 정의하고 있음.(사진에서만)
- yolov3-tiny.cfg - 작은 dataset일 때
- yolov3.cfg, yolov3.weights - 이 파일들을 가지고 yolo.h5를 생성함.

실행결과 

데이터의 부족으로 부드럽게 인지하지 못함.(labeling하는데 어려운 사진들과 크롤링, 스크레이핑 등으로 전체를 다운 받다보니까 쓸데없는 사진들도 포함되어 모았던 전체 데이터 1,500장 중 800장가량의 labeling에 성공했는데 대체로 10,000장 이상 필요함. 또한 YOLO 알고리즘 뿐만아니라 객체인식이라는 분야도 처음 도전해보는 분야임에 어떤데이터가 필요한 데이터이며 필요없는 데이터인지 구분하는데 한계가 있음)

## 2. (3) 카드 인식 후 신호 발송

카드의 외곽선이 인식되었을 때 1초길이의 삐 소리가 나도록 하기 위해 다음과 같은 코드를 작성하였다. 기존 코드에 위 코드를 이용하여 삐 소리가 나도록 시도해보았지만 여러 오류가 발생하였다. 카드가 인식되지 않을 때에도 삐 소리가 나거나 아예 삐 소리가 나지 않는 경우가 발생하였다. 카드의 외곽선이 검출되었을 때부터 카드가 영상에서 사라질 때까지만 삐 소리가 나도록 앞으로 계속해서 코드를 수정해 볼 예정이다.

## 3. 결론

여러 물체 중에서 특정 물체(credit_card)만을 인식하는 프로그램을 제작했다.
