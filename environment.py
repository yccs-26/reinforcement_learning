import time
import numpy as np
import tkinter as tk
from PIL import ImageTk, Image

PhotoImage = ImageTk.PhotoImage
UNIT = 40 # 픽셀 수
HEIGHT = 20  # 그리드 세로
WIDTH = 20  # 그리드 가로

np.random.seed(1)


class Env(tk.Tk):
    def __init__(self, render_speed=0.01):
        super(Env, self).__init__()
        self.render_speed=render_speed
        self.action_space = ['u', 'd', 'l', 'r']
        self.action_size = len(self.action_space)
        self.title('DeepSARSA')
        self.geometry('{0}x{1}'.format(HEIGHT * UNIT, HEIGHT * UNIT))
        self.shapes = self.load_images()
        self.canvas = self._build_canvas()
        self.counter = 0
        self.rewards = []
        self.goal = []
        #이동 장애물 설정
        self.set_reward([0, 1], -1)
        self.set_reward([2, 6], -1)
        self.set_reward([3, 3], -1)
        self.set_reward([7, 2], -1)
        self.set_reward([0, 1], -1)
        self.set_reward([12, 12], -1)
        self.set_reward([9, 18], -1)
        self.set_reward([17, 2], -1)
        self.set_reward([1, 9], -1)
        self.set_reward([11, 10], -1)

        # 고정 장애물 설정
        self.set_reward([3, 0], -2)
        self.set_reward([5, 5], -2)
        self.set_reward([1, 5], -2)
        self.set_reward([9, 0], -2)
        self.set_reward([15, 15], -2)
        self.set_reward([16, 4], -2)
        self.set_reward([1, 19], -2) 
        self.set_reward([4, 0], -2)
        self.set_reward([18, 17], -2)
        self.set_reward([18, 19], -2)
        self.set_reward([7, 16], -2) 
        # 목표 지점 설정
        self.set_reward([19, 19], 1)

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

        self.rewards = []
        self.goal = []
        # 캔버스에 이미지 추가
        x, y = UNIT/2, UNIT/2
        self.ship = canvas.create_image(x, y, image=self.shapes[0])

        canvas.pack()

        return canvas

    def load_images(self):
        ship = PhotoImage(
            Image.open("C:/Users/82108/OneDrive/바탕 화면/RL_py/Deepsa _sh/ship.jpg").resize((20, 20)))
        triangle = PhotoImage(
            Image.open("C:/Users/82108/OneDrive/바탕 화면/RL_py/triangle.png").resize((20, 20)))
        circle = PhotoImage(
            Image.open("C:/Users/82108/OneDrive/바탕 화면/RL_py/circle.png").resize((20, 20)))
        rock = PhotoImage(
            Image.open("C:/Users/82108/OneDrive/바탕 화면/RL_py/Deepsa _sh/rock.jpg").resize((20, 20)))

        return ship, triangle,circle,rock

    def reset_reward(self):

        for reward in self.rewards:
            self.canvas.delete(reward['figure'])

        self.rewards.clear()
        self.goal.clear()
        self.set_reward([0, 1], -1)
        self.set_reward([2, 6], -1)
        self.set_reward([3, 3], -1)
        self.set_reward([7, 2], -1)
        self.set_reward([0, 1], -1)
        self.set_reward([12, 12], -1)
        self.set_reward([9, 18], -1)
        self.set_reward([17, 2], -1)
        self.set_reward([1, 9], -1)
        self.set_reward([11, 10], -1)
        #고정 장애물 설정
        self.set_reward([3, 0], -2)
        self.set_reward([5, 5], -2)
       
        self.set_reward([1, 5], -2)
        self.set_reward([9, 0], -2)
        self.set_reward([15, 15], -2)
        self.set_reward([16, 4], -2)
        self.set_reward([1, 19], -2) 
        self.set_reward([4, 0], -2)
        self.set_reward([18, 17], -2)
        self.set_reward([18, 19], -2)
        
        self.set_reward([7, 16], -2) 
        # #goal
        self.set_reward([19, 19], 1)

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


        elif reward == -1:
            temp['direction'] = -1
            temp['reward'] = reward
            temp['figure'] = self.canvas.create_image((UNIT * x) + UNIT / 2,
                                                      (UNIT * y) + UNIT / 2,
                                                      image=self.shapes[1])
        
        elif reward == -2:
            temp['direction'] = -2
            temp['reward'] = reward
            temp['figure'] = self.canvas.create_image((UNIT * x) + UNIT / 2,
                                                      (UNIT * y) + UNIT / 2,
                                                      image=self.shapes[3])

        temp['coords'] = self.canvas.coords(temp['figure'])
        temp['state'] = state
        self.rewards.append(temp)

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
        x, y = self.canvas.coords(self.ship)
        self.canvas.move(self.ship, UNIT / 2 - x, UNIT / 2 - y)
        self.reset_reward()
        return self.get_state()

    def step(self, action):
        self.counter += 1
        self.render()

        if self.counter % 2 == 1:
            self.rewards = self.move_rewards()

        next_coords = self.move(self.ship, action)
        check = self.check_if_reward(self.coords_to_state(next_coords))
        done = check['if_goal']
        reward = check['rewards']

        self.canvas.tag_raise(self.ship)

        s_ = self.get_state()

        return s_, reward, done

    def get_state(self):

        location = self.coords_to_state(self.canvas.coords(self.ship))
        agent_x = location[0]
        agent_y = location[1]

        states = list()

        for reward in self.rewards:
            reward_location = reward['state']
            states.append(reward_location[0] - agent_x)
            states.append(reward_location[1] - agent_y)
            if reward['reward'] == -1:
                states.append(-1)
                states.append(reward['direction'])
            elif reward['reward'] == -2:
                states.append(-2)
                states.append(reward['direction'])
            else:
                states.append(1)

        return states

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

        if s[0] == (WIDTH - 1) * UNIT + UNIT / 2:
            target['direction'] = 1
        elif s[0] == UNIT / 2:
            target['direction'] = -1

        if target['direction'] == -1:
            base_action[0] += UNIT
        elif target['direction'] == 1:
            base_action[0] -= UNIT
        else:
            pass
            

        if (target['figure'] is not self.ship
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

        self.canvas.move(target, base_action[0], base_action[1])

        s_ = self.canvas.coords(target)

        return s_

    def render(self):
        # 게임 속도 조정
        time.sleep(self.render_speed)
        self.update()
