import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"]="1"




class contextual_bandit():
    def __init__(self) :
        self.state = 0
        # 밴딧들의 손잡이 목록을 작성. 각 밴딧은 각각 손잡이 4, 2, 1이 최적이다.
        # shape = (2,4)
        self.bandits = np.array([[0.2, 0, -0.0, -5],[0.1, -5, 1, 0.25],[-5, 5, 5, 5]]) 
        self.num_bandits = self.bandits.shape[0]
        self.num_actions = self.bandits.shape[1]


    def getBandit(self) :
        # 각각의 에피소드에 대해 랜덤함 상태를 반환
        self.state = np.random.randint(0,len(self.bandits))
        return self.state
    
    def pullArm(self, action) :
        # 랜덤한 수를 얻는다.
        bandit = self.bandits[self.state, action] 
        result = np.random.randn(1)

        if result > bandit :
            # 양의 보상을 반환한다.
            return 1
        else :
            # 음의 보상을 반환한다.
            return -1


class agent():
    def __init__(self, lr, s_size, a_size) :
        # 네트워크의 피드포워드 부분. 에이전트는 상태를 받아서 액션을 출력한다.
        self.state_in = tf.placeholder(shape=[1],dtype=tf.int32)
        state_in_OH = slim.one_hot_encoding(self.state_in, s_size)
        output = slim.fully_connected(state_in_OH, a_size, \
                biases_initializer=None, activation_fn=tf.nn.sigmoid, \
                weights_initializer=tf.ones_initializer())
        self.output = tf.reshape(output,[-1])
        self.chosen_action = tf.argmax(self.output,0)


        # 학습 과정을 구현한다.
        # 비용을 계산하기 위해 보상과 선택된 액션을 네트워크에 피드하고,
        # 네트워크를 업데이트하는 데에 이를 이용한다.
        self.reward_holder = tf.placeholder(shape=[1], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[1], dtype=tf.int32)
        self.responsible_weight = tf.slice(self.output, self.action_holder, [1])
        self.loss = -(tf.log(self.responsible_weight)*self.reward_holder)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate = lr)
        self.update = optimizer.minimize(self.loss)


# 텐서플로 그래프를 리셋한다.
tf.reset_default_graph()

# 밴딧을 로드한다.
cBandit = contextual_bandit()
# 에이전트를 로드한다.
myAgent = agent(lr=0.001, s_size=cBandit.num_bandits, a_size=cBandit.num_actions)
# 네트워크 내부를 들여다보기 위해 평가할 가중치 
weights = tf.trainable_variables()[0]

# 에이전트를 학습시킬 전체 에피소드 수 설정
total_episodes = 10000
# 밴딧에 대한 점수판을 0으로 설정
total_reward = np.zeros([cBandit.num_bandits, cBandit.num_actions])
# 랜덤한 액션을 취할 가능성 설정
e = 0.1




config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
# 텐서플로 그래프 론칭
with tf.Session(config = config) as sess :

    
    sess.run(tf.global_variables_initializer())


    i = 0
    while i < total_episodes :
        # 환경으로부터 상태 가져오기
        s = cBandit.getBandit()
        # 네트워크로부터 랜덤한 액션 또는 하나의 액션을 선택한다.
        if np.random.rand(1) < e :
            action = np.random.randint(cBandit.num_actions)

        else :
            action = sess.run(myAgent.chosen_action, feed_dict={myAgent.state_in:[s]})

        # 주어진 밴딧에 대해 액션을 취한 데 대한 보상을 얻는다.
        reward = cBandit.pullArm(action)


        # 네트워크를 업데이트한다.
        feed_dict={myAgent.reward_holder:[reward], \
                myAgent.action_holder:[action], \
                myAgent.state_in:[s]}

        _, ww = sess.run([myAgent.update, weights], feed_dict = feed_dict)


        # 보상의 총계 업데이트
        total_reward[s, action] += reward
        if i % 500 == 0 :
            print("Mean reward for each of the " + str(cBandit.num_bandits) + \
                    " bandit: " + str(np.mean(total_reward, axis=1)))

        i+=1


for a in range(cBandit.num_bandits) :
    print("The agent thinks action " + str(np.argmax(ww[a]+1)) + " for bandit " + \
            str(a+1) + " is the most promising.....")

    if np.argmax(ww[a]) == np.argmin(cBandit.bandits[a]) :
        print("... and it was right!")

    else :
        print("... and it was wrong!")









