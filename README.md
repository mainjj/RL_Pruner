# RL_Pruner
### Neural Network를 pruning 하기 위한 심층강화학습 기반 알고리즘 구현
  점점 커지는 신경망 규모로 인해 그에 따른 많은 resource 필요
  문제 해결을 위해 경량화 진행 요구됨

  본 프로젝트는 경량화 기법 중 pruning에 집중함
  
  Pruning: 중요도가 낮은 weight를 제거하여 파라미터 수를 줄이는 방법
  Pruning의 문제: 중요도가 낮은 weight를 사전에 알 수 없음, 시행작오가 필요함

  시행착오를 잘하는 RL를 활용하여 문제 해결
  Sparsity와 accuracy간의 균형을 맞추는 것이 목적인 RL 생성


### pruning결과
[ResNet50대상으로 test진행]
  Accuracy: 86.11% -> 83.26% 약 3%만큼 하락했지만
  parameter수: 23,528,522개 -> 5,687,226개 (약 76% 감소)
  기존 FLOPs 8.26에서 2.04로 줄어듬 (4배가량 빨라짐)
