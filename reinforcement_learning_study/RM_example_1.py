import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np


# 밴딧의 손잡이 목록을 작성한다.
# 현재 손잡이 4(인덱스틑 3)가 가장 자주 양의 보상을 제공하도록 설정되어 있다.

bandit_arms = [0.2, 0, -0.2, -2]
num_arms = len(bandit_arms)
def pullBandit(bandit) :
    # 랜덤한 값을 구한다.
    result = np.random.randn(1)
    if result > bandit :
        # 양의 보상을 반환한다.
        return 1
    else : 
        # 음의 보상을 반환한다.
        return -1

# 네트워크의 피드포워드 부분을 구현한다. 
weights = tf.Variable(tf.ones([num_arms]))
output = tf.nn.softmax(weights)

# 학습 과정을 구현한다.
# 보상과 선택된 애션을 네트워크에 피드해줌으로써 비용을 계산하고
# 비용을 이용해 네트워크를 업데이트 한다.
reward_holder = tf.placeholder(shape=[1], dtype=tf.float32)
action_holder = tf.placeholder(shape=[1], dtype=tf.int32)

responsible_output = tf.slice(output, action_holder,[1])
loss = -(tf.log(responsible_output)*reward_holder)
optimizer = tf.train.AdamOptimizer(learning_rate = 1e-3)
update = optimizer.minimize(loss)

# 에이전트를 학습시킬 총 에피소드의 수를 설정한다. 
total_episodes = 1000
# 밴딧 손잡이에 대한 점수판을 0으로 설정
total_reward = np.zeros(num_arms)

init = tf.global_variables_initializer()

# 텐서플로 그래프를 론칭한다.
with tf.Session() as sess :
    sess.run(init)
    i = 0
    while i <total_episodes :
        # 볼츠만 분포에 따라 액션 선택
        actions = sess.run(output)
        a = np.random.choice(actions, p = actions)
        action = np.argmax(actions == a)

        # 밴딧 손잡이 중 하나를 선택함으로써 보상을 받는다.
        reward = pullBandit(bandit_arms[action])

        # 네트워크를 업데이트한다.
        _, resp, ww, oo = sess.run([update, responsible_output, weights, output],\
                feed_dict={reward_holder : [reward],action_holder:[action]})
        
        # 보상 총계 업데이트
        total_reward[action] += reward
        if i % 50 == 0 :
            print("Running reward for the " + str(num_arms) +" arms of the bandit : "\
                    +str(total_reward))

        i+=1

print("\nThe agent thinks arm " + str(np.argmax(ww)+1) + " is the most promizing....")

print(ww)

print(oo)

if np.argmax(ww) == np.argmax(-np.array(bandit_arms)) :
    print("... and it was right!")
else :
    print("... and is was wrong!")



































