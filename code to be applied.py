import tkinter as tk
from PIL import Image, ImageTk

class GridApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TEST")
        
        self.canvas = tk.Canvas(self.root, width=250, height=250, bg='sky blue')
        self.canvas.grid(row=0, column=0, columnspan=6)
        
        self.square_size = 50
        self.draw_grid()
        
        # 보트 이미지 파일 경로 설정
        self.boat_image = Image.open("boat.png")
        self.boat_image = self.boat_image.resize((self.square_size, self.square_size), Image.LANCZOS)
        self.boat_photo = ImageTk.PhotoImage(self.boat_image)
        
        self.boat = self.canvas.create_image(0, 0, anchor='nw', image=self.boat_photo)
        
        # 항구 이미지 파일 경로 설정
        self.port_image = Image.open("port.png")
        self.port_image = self.port_image.resize((self.square_size, self.square_size), Image.LANCZOS)
        self.port_photo = ImageTk.PhotoImage(self.port_image)
        
        self.port = self.canvas.create_image(200, 200, anchor='nw', image=self.port_photo)

        # 암석 사진 추가
        self.stone_image_path = "stone.png"
        self.stone_image = Image.open(self.stone_image_path)
        self.stone_image = self.stone_image.resize((self.square_size, self.square_size), Image.LANCZOS)
        self.stone_photo = ImageTk.PhotoImage(self.stone_image)

        # 암석 위치 저장
        self.stone_positions = [(50, 50), (150, 150)]
        for pos in self.stone_positions:
            self.canvas.create_image(pos[0], pos[1], anchor='nw', image=self.stone_photo)

        # 산호 이미지 추가
        self.coral_image_path = "coral.png"
        self.coral_image = Image.open(self.coral_image_path)
        self.coral_image = self.coral_image.resize((self.square_size, self.square_size), Image.LANCZOS)
        self.coral_photo = ImageTk.PhotoImage(self.coral_image)

        # 산호 위치 저장
        self.coral_positions = [(150,50)]
        for pos in self.coral_positions:
            self.canvas.create_image(pos[0], pos[1], anchor='nw', image=self.coral_photo)

        # 물고기 이미지 추가
        self.fish_image_path = "fish.png"
        self.fish_image = Image.open(self.fish_image_path)
        self.fish_image = self.fish_image.resize((self.square_size, self.square_size), Image.LANCZOS)
        self.fish_photo = ImageTk.PhotoImage(self.fish_image)

        # 물고기 위치 저장
        self.fish_positions = [(50,150)]
        for pos in self.fish_positions:
            self.canvas.create_image(pos[0], pos[1], anchor='nw', image=self.fish_photo)

        # 방향 버튼 프레임 추가
        buttons_frame = tk.Frame(self.root)
        buttons_frame.grid(row=1, column=0, columnspan=6, pady=10)
        
        btn_up = tk.Button(buttons_frame, text="↑", command=self.move_up)
        btn_up.grid(row=0, column=1, padx=5)

        btn_down = tk.Button(buttons_frame, text="↓", command=self.move_down)
        btn_down.grid(row=1, column=1, padx=5)

        btn_left = tk.Button(buttons_frame, text="←", command=self.move_left)
        btn_left.grid(row=1, column=0, padx=5)

        btn_right = tk.Button(buttons_frame, text="→", command=self.move_right)
        btn_right.grid(row=1, column=2, padx=5)

        btn_diagonal_forward = tk.Button(buttons_frame, text="↘", command=self.move_diagonal_forward)
        btn_diagonal_forward.grid(row=1, column=3, padx=5)

        btn_diagonal_backward = tk.Button(buttons_frame, text="↙", command=self.move_diagonal_backward)
        btn_diagonal_backward.grid(row=1, column=4, padx=5)

    def is_collision(self, new_x, new_y):
        for pos in self.stone_positions:
            if new_x == pos[0] and new_y == pos[1]:
                return True
        return False

    def move_up(self, event=None):
        x1, y1, x2, y2 = self.canvas.bbox(self.boat)
        new_x, new_y = x1, y1 - self.square_size
        if y1 > 0 and not self.is_collision(new_x, new_y):
            self.canvas.move(self.boat, 0, -self.square_size)

    def move_down(self, event=None):
        x1, y1, x2, y2 = self.canvas.bbox(self.boat)
        new_x, new_y = x1, y1 + self.square_size
        if y2 < self.canvas.winfo_height() and not self.is_collision(new_x, new_y):
            self.canvas.move(self.boat, 0, self.square_size)

    def move_left(self, event=None):
        x1, y1, x2, y2 = self.canvas.bbox(self.boat)
        new_x, new_y = x1 - self.square_size, y1
        if x1 > 0 and not self.is_collision(new_x, new_y):
            self.canvas.move(self.boat, -self.square_size, 0)

    def move_right(self, event=None):
        x1, y1, x2, y2 = self.canvas.bbox(self.boat)
        new_x, new_y = x1 + self.square_size, y1
        if x2 < self.canvas.winfo_width() and not self.is_collision(new_x, new_y):
            self.canvas.move(self.boat, self.square_size, 0)

    def move_diagonal_forward(self):
        x1, y1, x2, y2 = self.canvas.bbox(self.boat)
        new_x, new_y = x1 + self.square_size, y1 + self.square_size
        if x2 < self.canvas.winfo_width() and y2 < self.canvas.winfo_height() and not self.is_collision(new_x, new_y):
            self.canvas.move(self.boat, self.square_size, self.square_size)

    def move_diagonal_backward(self):
        x1, y1, x2, y2 = self.canvas.bbox(self.boat)
        new_x, new_y = x1 - self.square_size, y1 - self.square_size
        if x1 > 0 and y1 > 0 and not self.is_collision(new_x, new_y):
            self.canvas.move(self.boat, -self.square_size, -self.square_size)

    def draw_grid(self):
        for i in range(0, 250, self.square_size):
            self.canvas.create_line([(i, 0), (i, 250)], tag='grid_line')
            self.canvas.create_line([(0, i), (250, i)], tag='grid_line')

if __name__ == "__main__":
    root = tk.Tk()
    app = GridApp(root)
    root.mainloop()
