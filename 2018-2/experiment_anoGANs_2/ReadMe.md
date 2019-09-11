# Fault Detection Project for OES code


- 작성 날짜 : 2019.9.9
- 프로젝트 기간 : 2018.9 ~ 2018.12
- 개발환경 :
  - Ubuntu 16.04.5 LTS
  - python 3.5.2
  - tensorflow 1.6.0
  - TATAN X (Pascal) 12GB
- 소개
  - 본 문서는 저의 개발 역량을 보여주는 것이 목적입니다. 따라서 이 코드는 제가 직접 작성한 것입니다.   
  - 본 코드의 네트워크 기본 구조는 [anoGANs](https://arxiv.org/pdf/1703.05921) 이용하며, loss 함수와 각각의 네트워크의 구조는 실제 논문 구조에서 수정하였습니다.
  - 네트워크의 목적은 정상으로 정의된 데이터만으로 학습을 한 후, socre 함수를 정의하고 이를 이용하여 정상데이터와 비정상데이터를 구분하는 것이 목적입니다.
  - OES data에 적용된 모델입니다.



- python code 모듈 구조
```
{directory}
    |-- anomaly_{data}.py : 데이터 전처리 작업
    |-- model.py : 전체 네트워크 정의
    |-- network.py : 개별 네트워크 정의
    |-- utility.py : 그래프 그리기 및 저장 기능 등
test{number}.py : main 실행문
```

- python code 모듈 간 관계
 ![모듈 간 관계 추상화](figure_1.png)
  - 각 모듈 간의 관계는 위의 벤 다이어그램과 같습니다. 서로 교집합이 있는 곳의 기능이 수정이 된다면, 두 모듈이 수정되야 됨을 의미합니다. 예를들면 utility.py과 model.py의 교집합에 포함된 기능은 설계한 모델의 시각화 관련 함수입니다. 이 함수는 지도학습 또는 비지도학습과 같이 모델마다 달라지는 기능입니다. 또한 model.py은 anomaly_{data}.py와 교집합되는 곳이 없습니다. 이 의미는 anomaly_{data}.py가 수정되도 model.py에 영향을 끼치지 못한다는 것을 의미합니다.


- 구현 중 어려웠던 점
  - [Fault Detection Project for MNIST code](https://github.com/kangheeyong/2018-1-Deep-Learing-pc1/tree/master/2018-2/experiment_anoGANs_1)에서 구현한 코드를 가지고 anomaly_{data}.py와 utility.py의 일부분을 수정했기 때문에 큰 어려움은 없었습니다.

- 관련 자료
  - [dirty code](https://github.com/kangheeyong/2018-1-Deep-Learing-pc1/tree/master/2018-2/for_19411)
  - [Fault Detection Project for MNIST code](https://github.com/kangheeyong/2018-1-Deep-Learing-pc1/tree/master/2018-2/experiment_anoGANs_1)
