# ML의 실용과 몇가지 팁 강의

## 어떻게 learning rate를 잘 정할까?
Lagrge learning rate : overshooting의 가능성을 키운다!
 -> 학습의 결과가 점점 이상해질 가능성이 생긴다.

Small learning rate : takes too long, stops at local minimum
 -> 가다가 중간에서 멈춘다!

Try several learning rates
 -> 처음엔 대채로 0.01로 시작하고
 -> learning 결과에 따라 조정을 해보자!

## Data (X) preprocessing for gradient descent
이상해질 가능성을 줄이기 위해 normalie를 한다!
original data -> zero-centered data -> normalized data

### Standardization


## Overfitting
학습을 통해 만든 모델이 학습데이터에 너무 치중해져버리게 되는것!
실제 테스트데이터에서는 잘 안맞게 되어버릴 가능성이 있다!

## Solutions for overtfitting
 - more training data
 - Reduce the number of features
 - Regularization ???(regularization strength?)
```python
l2reg = 0.001 * reduce_sum(tf.square(W))  <- regularization 으로 쓸수있음!
```
## Performance evaluation : is this good?
얼마나 훌룡한가? 어떻게 평가할까?
training(교과서)과 test(실전 문제) data sets을 나눠서 쓰자!
training set을 training과 validation set으로 나눠서 learning late와 regularization strenth를 튜닝하는 용도로 쓸수도있음!

## Online learning

## Accuracy
95~99% (이미지?)
