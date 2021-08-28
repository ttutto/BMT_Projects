# BMT_Projects
 
[���İ�(AlphaGo)](https://www.nature.com/articles/nature16961)�� [���İ� ����(AlphaGo Zero)](https://www.nature.com/articles/nature24270)�� ������� ����Ͽ� ���� �ΰ�����(AI)�� �����ϴ� ������Ʈ�Դϴ�.

�Ʒ� ���� ������Ʈ���� �ҽ��ڵ带 ���� Ȱ���Ͽ����ϴ�.
- [An implementation of the AlphaZero algorithm for Gomoku](https://github.com/junxiaosong/AlphaZero_Gomoku)
- [���ַ� �Ǻ�](https://blog.naver.com/dnpc7848/221506783416)

## ������Ʈ ���� ����
���� ������ �н�(train)�� ����(run) �� �������� ���е˴ϴ�.
- �н�(train)
  - ������(��ȭ�н�)�� Ȱ���� �н��� �ݺ��Ͽ� ��(model)�� �����մϴ�.
  - ���� ���� ����. (�̹� ������ ���� ����Ǿ� ����)
- ����(run)
  - �÷��̾�(���)�� �н��� ���� ������� �ϴ� AI(�ΰ�����)�� ������ �÷����� �� �ֽ��ϴ�.
  - :star:**�÷��� �ϴ� ���**:star:
 
```
git clone https://github.com/NohGod/BMT_Projects

cd BMT_Projects

python run.py
'''


## �����
����� �ֿ� ������� ������ ���� 3�����Դϴ�.
- ��å��(Policy Network)
  - ���İ�(AlphaGo)���� ���� ���������, ���� �������� ����(state)�� �Է¹޾� �� ��ġ�� ���� ����� ����մϴ�.
  - ����� ũ�ٴ� ���� �ڽ� �Ǵ� ��밡 �����ϱ� ���� ��ġ�� �ǹ��մϴ�.
- �ڰ� �뱹(self-play)�� ���� �н�
  - ���İ� ����(AlphaGo Zero)���� ���� ���������, �ڰ� �뱹�� ���ؼ� ������ �÷��� �����͸��� ����Ͽ� ��å���� �н��մϴ�.
  - �ΰ��� �÷��� �����͸� ���� ������� �ʾ�������, ���İ� ���δ� ���İ��� �پ ������ �����־����ϴ�.
- MCTS(Monte Carlo Tree Search : ����ī���� Ʈ�� Ž��) �˰���
  - ���İ�� ���İ� ���ο��� ���� ���������, �پ��� ����� ���� Ž���Ͽ� ���������� ���� ��ġ�� �����մϴ�.


## ���� �ڷ�
��ü���� ����� �ľ��ϴµ� �����մϴ�.
- [���İ� - ������Ű](https://namu.wiki/w/%EC%95%8C%ED%8C%8C%EA%B3%A0)
- [����(���ַ�) - ������Ű](https://namu.wiki/w/%EC%98%A4%EB%AA%A9?from=%EC%98%A4%EB%AA%A9%28%EA%B2%8C%EC%9E%84%29#s-2.3)

�̷������� ������ �ڷ���Դϴ�.
- [��ȭ�н��� ����](https://jeinalog.tistory.com/20)
- [DQN�� Policy Gradient�� ����](https://gist.github.com/ByungSunBae/56009ed6ea31bb91a236e67bcb3245a2)
- [DQN�� Policy Gradient�� ����(2)](https://dnddnjs.gitbooks.io/rl/content/numerical_methods.html)
- [Minimax �˰���� MCTS �˰���](https://shuuki4.wordpress.com/2016/03/11/alphago-alphago-pipeline-%ED%97%A4%EC%A7%91%EA%B8%B0/)
- [���İ� �� ����](https://blog.naver.com/sogangori/220668124217)
- [���İ� ����� ������ �˰��� �м�](https://brunch.co.kr/@justinleeanac/2)
- [���İ� ���� �м�(MCTS)](https://leekh7411.tistory.com/1?category=768501)
- [���İ� ���ο� �������ο� ���� �м�](https://jsideas.net/AlphaZero/)
