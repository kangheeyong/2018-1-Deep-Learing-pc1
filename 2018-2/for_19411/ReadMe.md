# Fault Detection Project about dirty code


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
  - 최종 code 작성 과정에서 작성된 dirty code입니다.
