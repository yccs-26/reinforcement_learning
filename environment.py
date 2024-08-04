import time
import numpy as np
import tkinter as tk  # 5x5 그리드 환경 구현하는 라이브러리
from tkinter import messagebox
from PIL import ImageTk, Image
import json

PhotoImage = ImageTk.PhotoImage
UNIT = 50  # 픽셀 수
HEIGHT = 5  # 그리드 세로
WIDTH = 5  # 그리드 가로

np.random.seed(1)


class Env(tk.Tk):
    def __init__(self, obstacles, goal_pos, agent_pos, render_speed=0.01):
        super(Env, self).__init__()
        self.render_speed=render_speed 
        #render_speed -프레임 간 대기 시간, 시각적 피드백
        self.action_space = ['u', 'd', 'l', 'r']
        self.action_size = len(self.action_space) 
        self.title('DeepSARSA')
        self.geometry('{0}x{1}'.format(HEIGHT * UNIT, HEIGHT * UNIT))
        self.shapes = self.load_images()
        self.canvas = self._build_canvas()
        self.counter = 0
        self.rewards = []
        self.goal = []
        self.obstacles = obstacles
        # 장애물 설정  - 이거를 나중에 유동적으로 만들면 선박 역할, 장애물의 수준에 따라서 reward 다르게 측정해도 될 것
        for obs in obstacles:
            self.set_reward(obs, -1)
        # self.set_reward([0, 1], -1)
        # self.set_reward([1, 2], -1)
        # self.set_reward([2, 3], -1)
        
        # 목표 지점 설정 
        self.set_reward(goal_pos, -1)
        # self.set_reward([4, 4], 1)
        self.set_agent_position(agent_pos)

    def _build_canvas(self):
        canvas = tk.Canvas(self, bg='white', 
                           height=HEIGHT * UNIT,
                           width=WIDTH * UNIT)
        #그리드 틀 만들준 것. 배경 흰 색에 높이와 너비 규정
        
        # 그리드 생성
        for c in range(0, WIDTH * UNIT, UNIT):  # 0~400 by 80 , WIDTH 만큼 반복
            x0, y0, x1, y1 = c, 0, c, HEIGHT * UNIT #UNIT = 50픽셀 
            canvas.create_line(x0, y0, x1, y1) #세로선 그리기
        for r in range(0, HEIGHT * UNIT, UNIT):  # 0~400 by 80, HEIGHT 만큼 반복
            x0, y0, x1, y1 = 0, r, HEIGHT * UNIT, r
            canvas.create_line(x0, y0, x1, y1) #가로선 그리기 

        self.rewards = [] #보상, goal -> list만 만들어놓고 init, reward관련 함수, 
        self.goal = []
        # 캔버스에 이미지 추가
        x, y = UNIT/2, UNIT/2
        self.rectangle = canvas.create_image(x, y, image=self.shapes[0]) #load_images 함수에서 0번째 인덱스에 있는 이미지 

        canvas.pack()
        # tkinter 라이브러리에 속한 함수 - GUI 위젯을 배치하는 방법 
        # 위젯을 창 안에 배치할 때 사용  options 파라미터 사용 가능(side:top, bottom, left, right), fill(none, x, y, both), expand, padx, pady)
        # 레이아웃 복잡해지면 grid, place 레이아웃 매니저 추천 

        return canvas

    def load_images(self):
        rectangle = PhotoImage(
            Image.open("1-grid-world/img/rectangle.png").resize((30, 30)))
        triangle = PhotoImage(
            Image.open("1-grid-world/img/triangle.png").resize((30, 30)))
        circle = PhotoImage(
            Image.open("1-grid-world/img/circle.png").resize((30, 30)))

        return rectangle, triangle, circle

    def reset_reward(self):

        for reward in self.rewards:
            self.canvas.delete(reward['figure'])

        self.rewards.clear()
        self.goal.clear()
        # self.set_reward([0, 1], -1)
        # self.set_reward([1, 2], -1)
        # self.set_reward([2, 3], -1)

        # #goal
        # self.set_reward([4, 4], 1)

    def set_reward(self, state, reward):
        state = [int(state[0]), int(state[1])]
        x = int(state[0])
        y = int(state[1])
        temp = {}
        if reward > 0:
            temp['reward'] = reward
            temp['figure'] = self.canvas.create_image((UNIT * x) + UNIT / 2,
                                                       (UNIT * y) + UNIT / 2,
                                                       image=self.shapes[2]) #figure가 들어갈 위치 

            self.goal.append(temp['figure']) #goal에 figure 추가 ; 보상이 0 이상 


        elif reward < 0: #피해야 할 대상 ; reward에 추가
            temp['direction'] = -1
            temp['reward'] = reward
            temp['figure'] = self.canvas.create_image((UNIT * x) + UNIT / 2,
                                                      (UNIT * y) + UNIT / 2,
                                                      image=self.shapes[1])

        temp['coords'] = self.canvas.coords(temp['figure'])
        #coords 메서드 : Tkinter canvas - 특정 캔버스 객체의 현재 좌표를 가져오거나 새로운 좌표로 객체를 이동시킨다. 
        temp['state'] = state
        self.rewards.append(temp) #temp 정보를 추가 -> 나중에 객체로써 불러들일 수 있다(coords, state, figure -- 포함)

    def set_agent_position(self, state):
        x, y = state
        self.canvas.coords(self.rectangle, (UNIT * x) + UNIT / 2, (UNIT * y) + UNIT / 2)
        
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
    #check_list -> if_goal, rewards : 해당 state가 목표 지점이 맞는지? 최종 보상이 맞는지? 

    def coords_to_state(self, coords): #coords의 x,y 좌표 return 
        x = int((coords[0] - UNIT / 2) / UNIT)
        y = int((coords[1] - UNIT / 2) / UNIT)
        return [x, y]

    def reset(self):    # agent의 위치, reward, figure, goal 초기화 
        self.update()
        time.sleep(0.5)
        x, y = self.canvas.coords(self.rectangle) #agent -> rectangle 
        self.canvas.move(self.rectangle, UNIT / 2 - x, UNIT / 2 - y)
        self.reset_reward()
        return self.get_state()

    def step(self, action):
        self.counter += 1
        self.render()

        if self.counter % 2 == 1:
            self.rewards = self.move_rewards()

        next_coords = self.move(self.rectangle, action) #agent가 action한 next_state의 좌표 
        check = self.check_if_reward(self.coords_to_state(next_coords)) # goal 도착했는지 check -> if_goal & rewards 있을 것 
        done = check['if_goal'] 
        reward = check['rewards']

        self.canvas.tag_raise(self.rectangle) #tag_raise : Tkinter - GUI ; 선택한 요소가 다른 요소들 위에 오도록 나타내는 함수 
        #-> agent가 다른 figure가 있는 곳에 오면 agent를 우선 표시

        s_ = self.get_state() #agent의 현재 위치 

        return s_, reward, done #agent의 위치, 보상, 목적지 도달 여부 

    def get_state(self):

        location = self.coords_to_state(self.canvas.coords(self.rectangle)) #agent의 x,y 좌표
        agent_x = location[0]
        agent_y = location[1]

        states = list() 

        for reward in self.rewards:
            reward_location = reward['state']
            states.append(reward_location[0] - agent_x)
            states.append(reward_location[1] - agent_y) # reward가 agent로부터 얼마나 떨어져 있는지
            if reward['reward'] < 0: # 장애물일 때
                states.append(-1)
                states.append(reward['direction']) # 피해야 하는 걸 나타내기 위해 -1과 보상의 방향을 states에 추가 
            else:
                states.append(1) # 목적지일 때 1로 표시

        return states

    def move_rewards(self):
        new_rewards = []
        for temp in self.rewards:
            if temp['reward'] == 1: # reward 목적지일 때 new_rewards에 temp 추가 (여기서의 템프는 임시 변수 템프) **?
                new_rewards.append(temp)
                continue
            temp['coords'] = self.move_const(temp)
            temp['state'] = self.coords_to_state(temp['coords'])
            new_rewards.append(temp)
        return new_rewards

    def move_const(self, target): # x축으로 값을 설정하면 후에 y축으로 파라미터를 바꾸기만 하면 되므로 축 한 개를 해결하는 함수. 

        s = self.canvas.coords(target['figure']) #좌표 가져오기

        base_action = np.array([0, 0]) #action에 대한 기본 값 = 이동 없음으로 설정 

        if s[0] == (WIDTH - 1) * UNIT + UNIT / 2: # 오른쪽 끝에 있으면 방향을 왼쪽으로 설정
            target['direction'] = 1
        elif s[0] == UNIT / 2:                    # 왼쯕 끝에 있으면 방향을 오른쪽으로 설정 
            target['direction'] = -1

        if target['direction'] == -1:
            base_action[0] += UNIT
        elif target['direction'] == 1:
            base_action[0] -= UNIT

        if (target['figure'] is not self.rectangle # target이 agent가 아니거나 캔버스 우측하단에 위치하면 이동하지 않는다. 왜? 안하는 게 아니라 못한다. 
           and s == [(WIDTH - 1) * UNIT, (HEIGHT - 1) * UNIT]):
            base_action = np.array([0, 0])

        self.canvas.move(target['figure'], base_action[0], base_action[1]) # target에 대해서 x축, y축으로 move() 

        s_ = self.canvas.coords(target['figure']) # 이동 후 좌표 반환 

        return s_

    def move(self, target, action):
        s = self.canvas.coords(target) # target 좌표 가져오기 -> 변수 s

        base_action = np.array([0, 0]) # base_action[0] - x좌표, base_action[1] - y좌표, 이동 X 

        if action == 0:  # 상
            if s[1] > UNIT: 
                base_action[1] -= UNIT # 개념적으로는 상(그리드에서는 밑)으로 갈 여유가 있으면 내린다.  
        elif action == 1:  # 하
            if s[1] < (HEIGHT - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:  # 우
            if s[0] < (WIDTH - 1) * UNIT: # x축은 그대로 받아들이면 되기에 오른쪽 끝이 아니라면 우측으로 한 칸 이동. 
                base_action[0] += UNIT
        elif action == 3:  # 좌
            if s[0] > UNIT:
                base_action[0] -= UNIT # 좌측으로 한 칸 이동 

        self.canvas.move(target, base_action[0], base_action[1]) # base_action의 x,y 축에 대한 결과가만큼 target 이동 

        s_ = self.canvas.coords(target) # s_에 target 좌표 저장 

        return s_

    def render(self):
        # 게임 속도 조정
        time.sleep(self.render_speed)
        # render_speed가 높을수록 진행이 느려진다. 
        self.update()

class SetUp:
    def __init__(self, root):
        self.root = root
        self.root.title("환경 설정")
        
        self.agent_position = None
        self.target_position = None
        self.obstacles = []
        # DB 이용해서 이전 에피소드의 장애물, 에이전트, 목적지 저장 
        # 저장하고 싶은 부분만 체크해서 남길 수 있는 방식도 괜찮을 것 같고
        # 즐겨찾기를 만들어서 자기가 조합할 수 있게끔 만들어도 좋을 것 같다.
        self.use_db = tk.BooleanVar() 
        
        self.setup_agent_target_frame()
    
    def setup_agent_target_frame(self):
        self.clear_frame()
        
        tk.Label(self.root, text="출발 지점과 목적지 설정").pack()
        
        tk.Label(self.root, text="Agent의 좌표 설정 (x, y) : ").pack()
        self.agent_x_entry = tk.Entry(self.root)
        self.agent_x_entry.pack()
        self.agent_y_entry = tk.Entry(self.root)
        self.agent_y_entry.pack()
        
        tk.Label(self.root, text="목적지 좌표 설정 (x, y) : ").pack()
        self.target_x_entry = tk.Entry(self.root)
        self.target_x_entry.pack()
        self.target_y_entry = tk.Entry(self.root)
        self.target_y_entry.pack()
        
        tk.Checkbutton(self.root, text="이전 에피소의 장애물들을 불러오시겠습니까?", variable=self.use_db).pack()
        tk.Button(self.root, text="장애물 설정하기", command=self.target_step).pack()
        
    def target_step(self):
        try:
            self.agent_position = (int(self.agent_x_entry.get()), int(self.agent_y_entry.get()))
            self.target_position = (int(self.target_x_entry.get()), int(self.target_y_entry.get()))
            
            if not self.validate_position(self.agent_position) or not self.validate_position(self.target_position):
                raise ValueError("유효하지 않은 좌표입니다")
            
            self.setup_obstacle_frame()
        except ValueError:
            messagebox.showerror("좌표 오류", "출발 지점과 목적지의 좌표를 다시 설정하십시오.")
            
    def setup_obstacle_frame(self):
        self.clear_frame()
        
        tk.Label(self.root, text="장애물 설정").pack()
        self.obstacle_positions = []
        
        tk.Label(self.root, text="장애물 좌표 설정 (x, y):").pack()
        self.obstacle_x_entry = tk.Entry(self.root)
        self.obstacle_x_entry.pack()
        self.obstacle_y_entry = tk.Entry(self.root)
        self.obstacle_y_entry.pack()
        
        tk.Button(self.root, text="장애물 추가하기", command=self.add_obstacle).pack()
        tk.Button(self.root, text="장애물 설정 완료", command=self.finish_setup).pack()
    
    def add_obstacle(self):
        try:
            pos = (int(self.obstacle_x_entry.get()), int(self.obstacle_y_entry.get()))
            
            if not self.validate_position(pos):
                raise ValueError("유효하지 않은 좌표")
            
            self.obstacles.append(pos)
            self.obstacle_positions.append(tk.Label(self.root, text=f"장애물 {len(self.obstacles)}: {pos}"))
            self.obstacle_positions[-1].pack()
        except ValueError:
            messagebox.showerror("좌표 오류", "장애물 좌표를 다시 설정하십시오.")
    
    def finish_setup(self):
        self.root.destroy()
        
        setup_values = {
            'agent_pos' : self.agent_position,
            'goal_pos' : self.target_position,
            'obstacles' : self.obstacles
        }
        
        with open('setup_values.json', 'w') as f:
            json.dump(setup_values, f, indent=4)
            
        env = Env(agent_pos=self.agent_position, goal_pos=self.target_position, obstacles=self.obstacles)
        env.mainloop()
    
    def validate_position(self, pos):
        x, y = pos
        return 0 <= x < WIDTH and 0 <= y < HEIGHT
    
    def clear_frame(self):
        for widget in self.root.winfo_children():
            widget.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = SetUp(root)
    root.mainloop()