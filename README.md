# BMT_Projects
 
[알파고(AlphaGo)](https://www.nature.com/articles/nature16961)와 [알파고 제로(AlphaGo Zero)](https://www.nature.com/articles/nature24270)의 방법론을 모방하여 오목 인공지능(AI)을 구현하는 프로젝트입니다.

아래 유사 프로젝트들의 소스코드를 적극 활용하였습니다.
- [An implementation of the AlphaZero algorithm for Gomoku](https://github.com/junxiaosong/AlphaZero_Gomoku)
- [렌주룰 판별](https://blog.naver.com/dnpc7848/221506783416)

## 프로젝트 실행 과정
실행 과정은 학습(train)과 실행(run) 두 과정으로 구분됩니다.
- 학습(train)
  - 딥러닝(강화학습)을 활용한 학습을 반복하여 모델(model)을 생성합니다.
  - 과정 생략 가능. (이미 생성된 모델이 저장되어 있음)
- 실행(run)
  - 플레이어(사람)는 학습된 모델을 기반으로 하는 AI(인공지능)와 오목을 플레이할 수 있습니다.
  - 상황에 따라 필요한 라이브러리는 설치해야 할 수 있습니다.

## 플레이 하는 방법
 Anaconda prompt 1
```
git clone https://github.com/NohGod/BMT_Projects

cd BMT_Projects

python run.py
```
Anaconda  prompt 2
```
pip install opencv-contrib-python
pip install Mediapipe
pip install autopy

cd BMT_Projects

python mouse.py
```


## 방법론
사용한 주요 방법론은 다음과 같이 3가지입니다.
- 정책망(Policy Network)
  - 알파고(AlphaGo)에서 사용된 방법론으로, 현재 오목판의 상태(state)를 입력받아 각 위치에 대한 기댓값을 계산합니다.
  - 기댓값이 크다는 것은 자신 또는 상대가 착수하기 좋은 위치를 의미합니다.
- 자가 대국(self-play)을 통한 학습
  - 알파고 제로(AlphaGo Zero)에서 사용된 방법론으로, 자가 대국을 통해서 생성한 플레이 데이터만을 사용하여 정책망을 학습합니다.
  - 인간의 플레이 데이터를 전혀 사용하지 않았음에도, 알파고 제로는 알파고보다 뛰어난 성능을 보여주었습니다.
- MCTS(Monte Carlo Tree Search : 몬테카를로 트리 탐색) 알고리즘
  - 알파고와 알파고 제로에서 사용된 방법론으로, 다양한 경우의 수를 탐색하여 최종적으로 착수 위치를 결정합니다.


## 참고 자료
전체적인 배경을 파악하는데 유용합니다.
- [알파고 - 나무위키](https://namu.wiki/w/%EC%95%8C%ED%8C%8C%EA%B3%A0)
- [오목(렌주룰) - 나무위키](https://namu.wiki/w/%EC%98%A4%EB%AA%A9?from=%EC%98%A4%EB%AA%A9%28%EA%B2%8C%EC%9E%84%29#s-2.3)

이론적으로 참고한 자료들입니다.
- [강화학습의 개념](https://jeinalog.tistory.com/20)
- [DQN과 Policy Gradient의 차이](https://gist.github.com/ByungSunBae/56009ed6ea31bb91a236e67bcb3245a2)
- [DQN과 Policy Gradient의 차이(2)](https://dnddnjs.gitbooks.io/rl/content/numerical_methods.html)
- [Minimax 알고리즘과 MCTS 알고리즘](https://shuuki4.wordpress.com/2016/03/11/alphago-alphago-pipeline-%ED%97%A4%EC%A7%91%EA%B8%B0/)
- [알파고 논문 번역](https://blog.naver.com/sogangori/220668124217)
- [알파고에 적용된 딥러닝 알고리즘 분석](https://brunch.co.kr/@justinleeanac/2)
- [알파고 제로 분석(MCTS)](https://leekh7411.tistory.com/1?category=768501)
- [알파고 제로와 알파제로에 대한 분석](https://jsideas.net/AlphaZero/)
