# Data Augmentation Generative Adversarial Networks(DAGANs) review
#
Generative data로 학습을 하면 성능이 좋아질까? 하는 호기심에 시작한 연구?지만, 이미 누군가가 같은 생각을 하고 이미 내가 전에한 실험보다 더 뛰어나게 정리해서 논문을 쓴걸 보면 나도 언제 저래볼까 하는 생각이 든다. 사실 이 논문의 경우는 어느정도 GANs에 대한 배경지식만 있다면 이 논문에서 제안한 구조를 이해를 하면 쉽게 이해할 수 있는 논문이라고 생각한다.
먼저 Conditional GANs과 Data Augmentation GANs의 차이를 분석한 후 실제 이 알고리즘을 구현하고 Conditional GANs으로 나온 결과와 DAGANs으로 나온 결과를 비교해보겠다.  
