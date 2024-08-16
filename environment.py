import time
import numpy as np
import tkinter as tk
from PIL import ImageTk, Image

PhotoImage = ImageTk.PhotoImage
UNIT = 20 # 픽셀 수
HEIGHT = 40  # 그리드 세로
WIDTH = 40   # 그리드 가로

np.random.seed(3)


class Env(tk.Tk):
    def __init__(self,render_speed):
        super(Env, self).__init__()
        self.render_speed=render_speed
        self.action_space = ['u', 'd', 'l', 'r']
        self.action_size = len(self.action_space)
        self.title('A2C')
        self.geometry('{0}x{1}'.format(HEIGHT * UNIT, HEIGHT * UNIT))
        self.shapes = self.load_images()
        self.canvas = self._build_canvas()
        self.counter = 0
        self.rewards = []
        self.goal = []
        # 장애물 설정
        self.set_reward([0,5],-3)
        self.set_reward([39,5],-3)
        self.set_reward([6,0],-3)
        self.set_reward([5,39],-3)
        
        self.set_reward([0,5],-3)
        self.set_reward([(UNIT-1),5],-3)
        self.set_reward([6,0],-3)
        self.set_reward([5,(UNIT-1)],-3)

        for i in range(30):
            self.set_reward([np.random.randint(1,(UNIT-2)),np.random.randint(1,(UNIT-1))],-3)

        for i in range(30):
            self.set_reward([np.random.randint(1,(UNIT-1)),np.random.randint(1,(UNIT-2))],-3)
        # 목표 지점 설정
        for j in range(2):
            for i in range(2):
                self.set_reward([(UNIT-2)+i,(UNIT-2)+j],1)
      
    def _build_canvas(self):
        canvas = tk.Canvas(self, bg='white',
                           height=HEIGHT * UNIT,
                           width=WIDTH * UNIT)
        # 그리드 생성
        for c in range(0, WIDTH * UNIT, UNIT):  # 0~400 by 80
            x0, y0, x1, y1 = c, 0, c, HEIGHT * UNIT
            canvas.create_line(x0, y0, x1, y1)
        for r in range(0, HEIGHT * UNIT, UNIT):  # 0~400 by 80
            x0, y0, x1, y1 = 0, r, HEIGHT * UNIT, r
            canvas.create_line(x0, y0, x1, y1)

        # 캔버스에 이미지 추가
        x, y = UNIT/2, UNIT/2
        self.rectangle = canvas.create_image(x, y, image=self.shapes[0])

        canvas.pack()

        return canvas

    def load_images(self):
        rectangle = PhotoImage(
            Image.open("../img/rectangle.png").resize((15, 15)))
        triangle = PhotoImage(
            Image.open("../img/triangle.png").resize((15, 15)))
        circle = PhotoImage(
            Image.open("../img/circle.png").resize((15, 15)))

        return rectangle, triangle, circle

    def reset_reward(self):

        for reward in self.rewards:
            self.canvas.delete(reward['figure'])

        self.rewards.clear()
        self.goal.clear()

        # 장애물 설정
        
        self.set_reward([0,5],-3)
        self.set_reward([39,5],-3)
        self.set_reward([6,0],-3)
        self.set_reward([5,39],-3)
        

        for i in range(30):
            self.set_reward([np.random.randint(1,(UNIT-2)),np.random.randint(1,(UNIT-1))],-3)

        for i in range(30):
            self.set_reward([np.random.randint(1,(UNIT-1)),np.random.randint(1,(UNIT-2))],-3)
        # 목표 지점 설정
        for j in range(2):
            for i in range(2):
                self.set_reward([(UNIT-2)+i,(UNIT-2)+j],1)
    # state-> list 형태로 input으로 받음 


    # new methods
    def check_if_reward(self, state):
        check_list = dict()
        check_list['if_goal'] = False
        rewards = 0

        for reward in self.rewards:
            if reward['state'] == state:
                rewards += reward['reward']
                if reward['reward'] == 1:
                    check_list['if_goal'] = True

        check_list['rewards'] = rewards

        return check_list

    def coords_to_state(self, coords):
        x = int((coords[0] - UNIT / 2) / UNIT)
        y = int((coords[1] - UNIT / 2) / UNIT)
        return [x, y]

    def reset(self):
        self.update()
        time.sleep(0.5)
        x, y = self.canvas.coords(self.rectangle)
        self.canvas.move(self.rectangle, UNIT / 2 - x, UNIT / 2 - y)
        self.reset_reward()
        return self.get_state()

    def step(self, action):
        self.counter += 1
        self.render()

        if self.counter % 2 == 1:
            self.rewards = self.move_rewards()

        next_coords = self.move(self.rectangle, action)
        check = self.check_if_reward(self.coords_to_state(next_coords))
        done = check['if_goal']
        reward = check['rewards']

        self.canvas.tag_raise(self.rectangle)

        s_ = self.get_state()

        return s_, reward, done

    def get_state(self):
        # agent 좌표 
        location = self.coords_to_state(self.canvas.coords(self.rectangle))
        # x,y좌표 -> list형에서 추출 canvas coords-> list형태로 return함 
        agent_x = location[0]
        agent_y = location[1]
        # state list추가. 
        states = list()
        for reward in self.rewards:
            reward_location = reward['state']
            if (reward_location[0]-agent_x)==1:
                states.append(reward_location[0] - agent_x)

            else:
                states.append(reward_location[0] - agent_x)
            if (reward_location[1]-agent_y)==1: 
                    states.append(reward_location[1] - agent_y)
            elif (reward_location[0]-agent_x)==2:
                    states.append(reward_location[1] - agent_y)
            elif (reward_location[0]-agent_x)==3:
                    states.append(reward_location[1] - agent_y)
            else:
                states.append(reward_location[1] - agent_y)
            if reward['reward'] < 0:
                states.append(-3)
                states.append(reward['direction'])
            else:
                states.append(1)

        return states
    
    def set_reward(self, state, reward):
        state = [int(state[0]), int(state[1])]
        x = int(state[0])
        y = int(state[1])
        temp = {}
        if reward > 0:
            temp['reward'] = reward
            temp['figure'] = self.canvas.create_image((UNIT * x) + UNIT / 2,
                                                       (UNIT * y) + UNIT / 2,
                                                       image=self.shapes[2])

            self.goal.append(temp['figure'])


        elif reward == -3:
            temp['direction'] = -3
            temp['reward'] = reward
            temp['figure'] = self.canvas.create_image((UNIT * x) + UNIT / 2,
                                                      (UNIT * y) + UNIT / 2,
                                                      image=self.shapes[1])

        temp['coords'] = self.canvas.coords(temp['figure'])
        temp['state'] = state
        self.rewards.append(temp)

    def move_rewards(self):
        new_rewards = []
        for temp in self.rewards:
            if temp['reward'] == 1:
                new_rewards.append(temp)
                continue
            temp['coords'] = self.move_const(temp)
            temp['state'] = self.coords_to_state(temp['coords'])
            new_rewards.append(temp)
            
        return new_rewards

    def move_const(self, target):

        s = self.canvas.coords(target['figure'])

        base_action = np.array([0, 0])
        # 그리드 월드 한계
        if s[0] == (WIDTH - 1) * UNIT + UNIT / 2:
            target['direction'] = 1
        elif s[0] == UNIT / 2:
            target['direction'] = -1

        if target['direction'] == -1:
            base_action[0] 
        elif target['direction'] == 1:
            base_action[0] 
        


        if (target['figure'] is not self.rectangle
           and s == [(WIDTH - 1) * UNIT, (HEIGHT - 1) * UNIT]):
            base_action = np.array([0, 0])

        self.canvas.move(target['figure'], base_action[0], base_action[1])

        s_ = self.canvas.coords(target['figure'])

        return s_

    def move(self, target, action):
        s = self.canvas.coords(target)

        base_action = np.array([0, 0])

        if action == 0:  # 상
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:  # 하
            if s[1] < (HEIGHT - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:  # 우
            if s[0] < (WIDTH - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:  # 좌
            if s[0] > UNIT:
                base_action[0] -= UNIT
        # 객체 이동시키기 
        self.canvas.move(target, base_action[0], base_action[1])
        # target 이동 후 새로운 좌표 상태 
        s_ = self.canvas.coords(target)

        return s_

    def render(self):
        # 게임 속도 조정
        time.sleep(self.render_speed)
        self.update()
    
#env=Env(render_speed=0.01)
#env.mainloop()