import pylab
import numpy as np
from environment import Env
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomUniform


# 정책 신경망과 가치 신경망 생성
class A2C(tf.keras.Model):
    # sequential 사용이랑 차이가 먼지 공부해야함 
    def __init__(self, action_size,state_size):
        super(A2C,self).__init__()
        # 정책 신경망
        self.actor_fc1 = Dense(state_size, activation='relu')
        self.actor_fc2 = Dense(50, activation='relu')
        self.actor_fc3 = Dense(40, activation='relu')
        self.actor_fc4 = Dense(30, activation='relu')
        self.actor_fc5 = Dense(20, activation='relu')
        self.actor_fc6 = Dense(10, activation='relu')
        # 모든 합이 1 이 되어야해서 softmax함수를 사용합니다. 
        self.actor_out = Dense(action_size, activation='softmax', kernel_initializer=RandomUniform(-1e-3, 1e-3))
        # 가치 신경망 
        self.critic_fc1 = Dense(state_size, activation='relu')
        self.critic_fc2 = Dense(50, activation='relu')
        self.critic_fc3 = Dense(40, activation='relu')
        self.critic_fc4 = Dense(30, activation='relu')
        self.critic_fc5 = Dense(20, activation='relu')
        self.critic_fc6 = Dense(10, activation='relu')
        self.critic_out = Dense(1, kernel_initializer=RandomUniform(-1e-3, 1e-3))
        
    def call(self, x):
        #정책 신경망 대입
        actor_x = self.actor_fc1(x)
        actor_x = self.actor_fc2(actor_x)
        actor_x = self.actor_fc3(actor_x)
        actor_x = self.actor_fc4(actor_x)
        actor_x = self.actor_fc5(actor_x)
        actor_x = self.actor_fc6(actor_x)
        policy = self.actor_out(actor_x)
        #가치 신경망 대입
        critic_x = self.critic_fc1(x)
        critic_x = self.critic_fc2(critic_x)
        critic_x = self.critic_fc3(critic_x)
        critic_x = self.critic_fc4(critic_x)
        critic_x = self.critic_fc5(critic_x)
        critic_x = self.critic_fc6(critic_x)
        value = self.critic_out(critic_x)
        return policy, value

    # 그리드월드 A2C 에이전트
class A2CAgent:
    def __init__(self, state_size, action_size):
        # agent attr으로 상태의 크기와 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size

        # A2C 하이퍼 파라메터
        # 이 부분 조정 필요 
        self.discount_factor = 0.8
        self.learning_rate = 0.0001

        # 정책신경망과 가치신경망 생성
        # CLASS A2C에 self action size 대입 
        # self.model->정책 신경망임 A2C:attribute(actionsize) 함수 대입
   
        self.model = A2C(self.action_size,self.state_size)
        #self.model.load_weights('save_model/trained/model')
        # 최적화 알고리즘 설정, 미분값이 너무 커지는 현상을 막기 위해 clipnorm 설정
        self.optimizer = Adam(lr=self.learning_rate, clipnorm=3.0)
        #state,action,reward 보상 list 
        self.states, self.actions, self.rewards = [], [], []
        
    # 정책신경망의 출력을 받아 확률적으로 행동을 선택
    # self.model->정책 신경망 A2C:attribute(actionsize) 함수 대입
    def get_action(self, state):
        #python에서 버리는 값 _. value= 버리는 거. 
        policy, _ = self.model(state)
        policy = np.array(policy[0])
        return np.random.choice(self.action_size, 1, p=policy)[0]

    # 각 타임스텝마다 정책신경망과 가치신경망을 업데이트
    # reinforce에서 반환값을 없앴습니다. Td방식 채용으로 반환값을 가치신경망 대체 
    def train_model(self, state, action, reward, next_state, done):
        # 학습 파라메터: 
        model_params = self.model.trainable_variables
        #Returns all variables created with trainable=True.
        with tf.GradientTape() as tape:
            policy, value = self.model(state)
            # model return=policy, value-> policy 값만 가져오기 . 
            _, next_value = self.model(next_state)
            # model return=policy, value-> value 값만 가져오기 . 
            target = reward + (1 - done) * self.discount_factor * next_value[0]

            # 정책 신경망 오류 함수 구하기
            one_hot_action = tf.one_hot([action], self.action_size)
            action_prob = tf.reduce_sum(one_hot_action * policy, axis=1)
            #나블라 log로 표현
            cross_entropy = - tf.math.log(action_prob + 1e-5)
            advantage = tf.stop_gradient(target - value[0])
            actor_loss = tf.reduce_mean(cross_entropy * advantage)

            # 가치 신경망 오류 함수 구하기
            critic_loss = 0.5 * tf.square(tf.stop_gradient(target) - value[0])
            critic_loss = tf.reduce_mean(critic_loss)

            # 하나의 오류 함수로 만들기# 
            loss = 0.2 * actor_loss + critic_loss

        # 오류함수를 줄이는 방향으로 모델 업데이트
        grads = tape.gradient(loss, model_params)
        self.optimizer.apply_gradients(zip(grads, model_params))
        self.states, self.actions, self.rewards = [], [], []
        return np.array(loss)
 
if __name__ == "__main__":
    # 환경과 에이전트 생성
    env = Env(render_speed=0.000001)    
    state_size = len(env.get_state())
    action_space = [0, 1, 2, 3,4 ]
    action_size = len(action_space)
    
    agent = A2CAgent(state_size, action_size)

    scores, episodes = [], []
    score_avg = 0
    state=env.get_state()
    EPISODES = 2000
    #학습 횟수 설정입니다. 
    for e in range(EPISODES):
        done = False
        score = 0
        # loss 값 list
        loss_list = []
    
        # numpy로 state 집합의 배열을 1, state_size만큼의 행렬로 변환 
        state = np.reshape(state, [1, state_size])
        # 끝날때까지: 
        while not done:
            # 현재 상태에 대한 행동 선택(POLICY)
            # state=> get action을 신경망 모델에서 softmax로 추출 
            action = agent.get_action(state)
            # 선택한 행동으로 환경에서 한 타임스텝 진행 후 샘플 수집
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, len(next_state)])
            #처벌 
            reward=reward-0.5
            #print (score)
            score += reward
            
            if score<=-600: 
                reward=reward-100
                score += reward
                done = True
            # 매 타임스텝마다 학습
            loss = agent.train_model(state, action, reward, next_state, done)
            loss_list.append(loss)
            state = next_state
            
            if done:
                # 에피소드마다 학습 결과 출력
                state = env.reset()
                score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
                print("episode: {:3d} | score avg: {:3.2f} | loss: {:.3f}".format(
                      e, score_avg, np.mean(loss_list)))
                print (score)
            
                # 에피소드마다 학습 결과 그래프로 저장
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.xlabel("episode")
                pylab.ylabel("average score")
                pylab.savefig("./save_graph/graph.png")


        # 50 에피소드마다 모델 저장
        if e % 50 == 0:
           agent.model.save_weights('save_model/model', save_format='tf')
