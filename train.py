import copy # 객체의 얕은 복사, 깊은 복사 기능 제공 - agent의 상태나 환경을 복사
import pylab # matplotlib의 일부로 그래프 그리기와 같은 시각적 표현 도와준다. 
import random #난수 생성 모듈 - agent의 행동 무작위 선택 & 환경 무작위 설정 
import numpy as np # 수치 계산 - 상태, 행동, 보상 등을 배열로 관리하는데 사용 
from environment import Env # Env - agent가 상호작용하는 환경 정의 - agent가 어떻게 행동할지, 보상 어떻게 결정할지
from environment import SetUp
import tensorflow as tf # 머신러닝 라이브러리 - 계산 그래프로 복잡한 연산 
from keras.layers import Dense # tensorflow 위에서 활동하는 고수준 신경망 API - Dense : 완전 연결 레이어(fully connected layer)를 구현 
from keras.optimizers import Adam # Adam - 최적화 알고리즘의 한 종류 - 신경망의 가중치 업데이트할 때 사용
from tkinter import Tk
import json

# 딥살사 인공신경망
class DeepSARSA(tf.keras.Model): # tensorflow의 keras API 사용하여 모델링 수행 
    def __init__(self, action_size): # action_size - action의 개수 -> 출력층의 노드 수 결정 
        super(DeepSARSA, self).__init__() # tf.keras.model에서 받은 속성, 메서드 초기화 
        # super을 사용하지 않으면 제대로 초기화되지 않을 수 있다. (모델의 구조, 가중치) -> 못했을 경우 결과가 다르게 나올 수 있기에 
        # 확실하도록 super 함수를 이용하여 상속받은 클래스인 tf.keras.model의 초기화 로직 실행 
        self.fc1 = Dense(30, activation='relu')
        self.fc2 = Dense(30, activation='relu')
        self.fc_out = Dense(action_size)
        # fc1, fc2, fc_out -> 30개의 노드를 가진 2개의 은닉층과 1개의 출력층 & 은닉층에서는 ReLU 활성화 함수 사용
        
        # ReLU 활성화 함수란?(Rectified Linear Unit)
        # 선형성 - 계산이 빠르고 학습 속도 상향 > 이로 인해 기울기 소실 문제 완화
        # 비선형성 - 음수 입력에 대해 0 출력 > 복잡한 패턴과 함수 학습할 수 있도록
        # 효율성 - 계산 간단 > 계산 비용이 낮기에 모델 학습 시간 낮다.
        # 기울기 = 1 > 기울기 사라지는 현상 줄일 수 있다.
        # 단점으로는 죽은 ReLU : 특정 뉴런이 0만 출력 < 학습률 너무 높을 때 발생 
        # 보완책 -> Leaky ReLU
        
    def call(self, x): #모델을 호출할 때 쓰는 함수 
        x = self.fc1(x)
        x = self.fc2(x) # x - fc1, fc2를 거친 입력 데이터 
        q = self.fc_out(x) # 예측된 Q 값인 q 반환
        return q

        # 예측된 Q 값이란? : 각 layer을 통과하면서, 신경망은 주어진 상태에 대한 각 행동의 Q 값(action을 취했을 때 얻을 수 있는 예상 보상)을 예측
        # 이를 이용하여 agent가 어떤 action을 할 지 결정 


# 그리드월드 예제에서의 딥살사 에이전트
class DeepSARSAgent:
    def __init__(self, state_size, action_size):
        # 상태의 크기와 행동의 크기 정의 > 모델이 처리할 입력과 출력의 차원 결정(차원 = 행동의 가짓 수) - 4개 
        self.state_size = state_size
        self.action_size = action_size
        
        # 딥살사 하이퍼 파라메터
        self.discount_factor = 0.99 #할인 인자 - 보상의 미래 가치를 얼마나 중요시할지
        self.learning_rate = 0.001 # 학습률 - 모델 학습 속도
        # 탐욕 정책을 위한 epsilon 값 - 탐색과 활용의 균형 조절 
        self.epsilon = 1.   # agent가 무작위로 행동을 선택할 확률 -> 1로 설정 : 탐험을 최대화 - 환경에 대한 이해를 위해서
        self.epsilon_decay = .9999 # 시간이 지남에 따라 agent의 탐험을 줄이고 학습한 지식을 이용해 최적의 행동을 하도록 유도
        self.epsilon_min = 0.01 # 입실론의 최솟값 - 학습이 많이 되어도 일정 수준의 탐험을 진행하도록 > local optima(지역 최적해)에 빠지지 않게 하려고 
        self.model = DeepSARSA(self.action_size) # 딥살사 모델 > 상태를 입력으로 받아 각 행동에 대한 가치 예측 
        self.optimizer = Adam(lr=self.learning_rate) # DeepSARSA 모델, Adam 초기화 

    # 입실론 탐욕 정책으로 행동 선택
    def get_action(self, state):
        if np.random.rand() <= self.epsilon: # 무작위로 행동 선택 -> 탐색 : 입실론 값에 따라 무작위 행동인지 최적 행동인지 선택 
            return random.randrange(self.action_size)
        else: # 모델의 예측을 바탕으로 최적의 행동 선택 -> 활용 
            q_values = self.model(state) 
            return np.argmax(q_values[0])
    #왜 입실론 값에 따라 탐색과 활용 중 하나를 고르는가? 시간이 지남에 따라서 학습을 이용한 최적 선택을 유도하려고 

    # <s, a, r, s', a'>의 샘플로부터 모델 업데이트
    def train_model(self, state, action, reward, next_state, next_action, done):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay # 입실론 값 감소 > 갈수록 최적의 행동을 선택하도록 

        # 학습 파라메터
        model_params = self.model.trainable_variables # tensorflow에 있는 파라미터
        # 가중치 : layer의 뉴런들 사이 연결 강도(초기 무작위),  바이어스 : 각 레이어의 뉴런에 추가되는 고정값 > 뉴런의 활성화 조정 (학습 중 최적화)
        # Dense Layer(완전 연결 레이어) -> 각 연결에 대한 가중치와 각 레이어에 대한 바이어스 가진다. 
        with tf.GradientTape() as tape: # 손실함수에 대한 그래디언트 계산 -> 모델의 가중치 업데이트 
            # 파라미터의 그래디언트(미분값) 계산 위해 사용되는 API 
            # tape에 그래디언트 기록하는 범위 지정하고 기록 
            tape.watch(model_params) # model_params에 있는 텐서들에 대한 연산 주시(tf.Variable - 자동주시, tf.Tensor로 선언된 텐서 - 수동 주시)
            # model_params가 variable인지 tensor인지 모르니까 명시적으로 설정 
            predict = self.model(state)[0] # 주어진 상태에 대한 예측 
            one_hot_action = tf.one_hot([action], self.action_size) # action: 선택된 행동(상하좌우), action_size : 가능한 행동의 수
            # one_hot 인코딩이란? 
            # 선택된 행동을 제외한 모든 위치의 값을 0으로 설정, 선택된 행동만 1로 표시
            predict = tf.reduce_sum(one_hot_action * predict, axis=1)
            # one_hot 인코딩된 행동 X 예측 확률 > 선택된 행동에 대한 예측 확률만 추출. > 손실함수 계산할 때 사용하여 선택된 행동에 대한 예측의 정확성 평가 

            # done = True 일 경우 에피소드가 끝나서 다음 상태가 없음
            next_q = self.model(next_state)[0][next_action] 
            # 다음 상태에서 agent가 취할 수 있는 행동에 대한 예상 가치 Q를 모델을 이용하여 예측한다. 
            # [0] : 하나의 상태에 대해서만 예측 하기에 첫 번째 요소만을 선택하는 것이다. / [next_action] : 해당 벡터에서 next action에 해당하는 행동의 예상 가치 선택
            # self.model(next_state) : next state에 대해 모델이 예측한 모든 가능한 행동에 대해서 Q 값을 벡터 형태로 반환한다. 가능한 행동 4개이면 4개의 Q값을 포함하는 벡터를 출력
            # 첫 번째 예측 결과 벡터에서 next_action에 해당하는 인덱스의 Q 값을 선택 > next_action이 2이면 예측된 Q값 벡터에서 세 번째 원소 선택 
            # 모델이 예측한 Q 값 벡터에서 next_aciton의 값을 인덱스로 값을 선택하는 것이다. 
            target = reward + (1 - done) * self.discount_factor * next_q
            # 현재 행동의 타겟 Q 값 계산 : reward = 현재 행동에 대한 보상, done = episode 끝났는지 확인 여부, 할인 계수 * 다음 상태의 예상 가치 = 미래 가치에 대한 현재 가치의 중요도 
            # episode가 끝났으면 1-done=0 -> 미래 보상 = 0 이므로 현재 받은 보상만 반영
            
            # MSE 오류 함수 계산(손실함수) : 예측된 Q 값과 타겟 Q 값 사이의 평균 제곱 오차(MSE : 회귀 문제에서 자주 사용) 
            loss = tf.reduce_mean(tf.square(target - predict)) # 예측 오차의 제곱의 평균값
            # 손실함수 - 모델이 학습하는 동안 최소화돼야 한다. : 모델의 정확도 평가 지표 
            # tf.reduce_mean : 주어진 값들의 평균 계산 -> 하나의 손실값 얻는다.
            
            # MSE 손실함수 - 회귀 문제에 자주 사용하는 이유
            # 예측값과 실제값 차이가 크면 손실 크게 증가 -> 정확한 예측 유도
            # 큰 오차에 더 많은 가중치 부여 -> 에측 정확도 향상
            # 경사 하강법(gradient descent)과 같이 최적화 알고리즘 사용 > 최적화 용이 <- 파라미터에 대해 미분이 가능한 연속 함수 
            # 회귀 문제 = 연속적인 값을 예측하는 문제 < 미분가능  - 연속하는 성질 보일 수 있다.
            # 과적합 방지하는 데 도움된다. < 특정 데이터 과도하게 적용 시 큰 패널티 부여 
        
        # 오류함수를 줄이는 방향으로 모델 업데이트(각 batch, epoch마다 반복)
        grads = tape.gradient(loss, model_params) # 손실 함수에 대한 모델 파라미터의 기울기 계산 (tape : tf.gradient의 변수 - 연산 과정 기록, 이를 통해 미분 계산 - 기울기 get)
        # grads > 모델 어느 방향으로 얼마나 조정해야 손실 줄일 수 있는지 알 수 있다. 
        self.optimizer.apply_gradients(zip(grads, model_params)) # apply_gradients : 기울기와 파라미터 입력 받아 optimizer 규칙(Adam)에 따라 파라미터 업데이트 (zip으로 grads, 파라미터 묶어준다.)
        # optimizer - Adam의 인스턴스 : 최적화를 위해 사용 
        


if __name__ == "__main__": # 이 파일 직접 실행할 때만 작동하도록 하는 블록 
    # 환경과 에이전트 생성
    with open('setup_values.json', 'r') as f:
        setup_values = json.load(f)
        
    env = Env(
        obstacles=setup_values['obstacles'],
        goal_pos=setup_values['goal_pos'],
        agent_pos=setup_values['agent_pos'],
        render_speed=0.01) # Env()를 env 객체로 생성
    root = Tk()
    setup = SetUp(root)
    state_size = 15 
    action_space = [0, 1, 2, 3, 4]  # 가능한 action의 집합 
    action_size = len(action_space) # 가능한 행동의 수 - 5
    agent = DeepSARSAgent(state_size, action_size) 
    
    scores, episodes = [], [] # 총 점수 ,에피소드 번호 저장하기 위해 선언, 초기화

    EPISODES = 1000
    for e in range(EPISODES):
        done = False
        score = 0
        # env 초기화
        state = env.reset()
        state = np.reshape(state, [1, state_size]) # state 배열을 재구성, 첫 번째 차원에 1(batch 크기), 두 번째 차원에 state_size

        while not done: #아직 목적지 도달 못한 상태 
            # 현재 상태에 대한 행동 선택
            action = agent.get_action(state)

            # 선택한 행동으로 환경에서 한 타임스텝 진행 후 샘플 수집
            next_state, reward, done = env.step(action) 
            next_state = np.reshape(next_state, [1, state_size]) 
            next_action = agent.get_action(next_state)

            # 샘플로 모델 학습
            agent.train_model(state, action, reward, next_state, 
                                next_action, done)
            score += reward
            state = next_state

            if done: # dest 도달 완료 
                # 에피소드마다 학습 결과 출력
                print("episode: {:3d} | score: {:3d} | epsilon: {:.3f}".format(
                      e, score, agent.epsilon))

                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b') #blue, episode-x축, scores-y축
                pylab.xlabel("episode") 
                pylab.ylabel("score")
                pylab.savefig("graph.png")

        # 100 에피소드마다 모델 저장
        if e % 100 == 0:
            agent.model.save_weights('save_model/model', save_format='tf')