import random
import sys
import pygame
import networkx as nx
import matplotlib.pyplot as plt
import csv
import os
from pygame import gfxdraw

# 初始化pygame
pygame.init()
pygame.font.init()

# 颜色定义
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
DARK_GRAY = (100, 100, 100)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
ORANGE = (255, 165, 0)
YELLOW = (255, 255, 0)
PINK = (255, 192, 203)

# 字体设置
# 确保字体文件在相同目录下
FONT_SMALL = pygame.font.Font("msyhl.ttc", 14)
FONT_MEDIUM = pygame.font.Font("msyhl.ttc", 18)
FONT_LARGE = pygame.font.Font("msyhl.ttc", 24)

class Maze:
    """迷宫生成类"""
    def __init__(self, rows, columns):
        """初始化迷宫参数
        Args:
            rows: 迷宫行数
            columns: 迷宫列数
        """
        self.ROWS = rows
        self.COLUMNS = columns
        # 定义生成迷宫时的移动方向（右、下、左、上）
        self.x = [0, 2, 0, -2]  # x方向移动步长
        self.y = [2, 0, -2, 0]  # y方向移动步长
        # 定义寻路时的移动方向
        self.px = [0, 1, 0, -1]  # 寻路x方向移动步长
        self.py = [1, 0, -1, 0]  # 寻路y方向移动步长
        self.to_be_selected = []  # 待选节点列表
        self.random_selectB = []  # 随机选择的B点列表
        self.path_list = []      # 路径列表
        # 初始化访问标记矩阵
        self.isvisit = [[0 for _ in range(columns)] for _ in range(rows)]
        self.isvisit[1][1] = 1  # 起点已访问
        self.matrix = self.matrix_init(rows, columns)  # 初始化迷宫矩阵
        self.start = [1, 1]  # 设置起点
        self.put_node_in_to_be_selected(self.start)  # 将起点加入待选列表
        self.path_list.append(self.start)  # 将起点加入路径
        self.is_generate = False  # 标记迷宫是否生成完成
        self.generation_complete = False  # 标记迷宫生成过程是否完成
        self.path_finding_complete = False  # 标记路径查找是否完成

    def matrix_init(self, r, c):
        """初始化迷宫矩阵
        Args:
            r: 行数
            c: 列数
        Returns:
            初始化后的迷宫矩阵（1表示墙，0表示通道）
        """
        matrix = [[1 for _ in range(c)] for _ in range(r)]
        matrix[1][1] = 0  # 设置起点
        return matrix

    def put_node_in_to_be_selected(self, node):
        """将当前节点的可选邻节点加入待选列表"""
        for i in range(4):
            xx = node[0] + self.x[i]
            yy = node[1] + self.y[i]
            # 检查节点是否在边界内且未被访问过
            if (
                0 < xx < self.ROWS
                and 0 < yy < self.COLUMNS
                and [xx, yy] not in self.to_be_selected
                and self.matrix[xx][yy] == 1
            ):
                self.to_be_selected.append([xx, yy])

    def random_B(self, node):
        """随机选择与当前节点相连的已访问节点
        Args:
            node: 当前节点
        Returns:
            随机选择的已访问节点
        """
        self.random_selectB.clear()
        for i in range(4):
            xx = node[0] + self.x[i]
            yy = node[1] + self.y[i]
            if 0 < xx < self.ROWS and 0 < yy < self.COLUMNS and self.matrix[xx][yy] == 0:
                self.random_selectB.append([xx, yy])
        if len(self.random_selectB) > 0:
            rand_B = random.randint(0, len(self.random_selectB) - 1)
            return self.random_selectB[rand_B]
        return None

    def matrix_generate(self):
        """生成迷宫结构"""
        # 第一阶段：生成迷宫主体结构
        if len(self.to_be_selected) > 0:
            # 随机选择一个待访问节点
            rand_s = random.randint(0, len(self.to_be_selected) - 1)
            select_nodeA = self.to_be_selected[rand_s]
            # 随机选择一个已访问节点与之相连
            selectB = self.random_B(select_nodeA)
            if selectB is not None:
                self.matrix[select_nodeA[0]][select_nodeA[1]] = 0
                # 打通两点之间的墙
                mid_x = int((select_nodeA[0] + selectB[0]) / 2)
                mid_y = int((select_nodeA[1] + selectB[1]) / 2)
                self.matrix[mid_x][mid_y] = 0
                self.put_node_in_to_be_selected(select_nodeA)
                self.to_be_selected.remove(select_nodeA)
        
        # 第二阶段：寻找从起点到终点的路径
        elif len(self.path_list) > 0 and not self.is_generate:
            self.matrix[self.ROWS - 1][self.COLUMNS - 1] = 3  # 标记终点
            l = len(self.path_list) - 1
            n = self.path_list[l]
            # 如果到达终点，结束寻路
            if n[0] == self.ROWS - 1 and n[1] == self.COLUMNS - 1:
                self.is_generate = True  # 标记迷宫生成完成
                self.generation_complete = True
                return
            # 尝试四个方向移动
            for i in range(4):
                xx = n[0] + self.px[i]
                yy = n[1] + self.py[i]
                # 检查移动是否有效
                if (
                    0 < xx < self.ROWS
                    and 0 < yy < self.COLUMNS
                    and (self.matrix[xx][yy] == 0 or self.matrix[xx][yy] == 3)
                    and self.isvisit[xx][yy] == 0
                ):
                    self.isvisit[xx][yy] = 1
                    self.matrix[n[0]][n[1]] = 2  # 标记路径
                    tmp = [xx, yy]
                    self.path_list.append(tmp)
                    break
                # 如果四个方向都无法移动，回溯
                elif i == 3:
                    self.matrix[n[0]][n[1]] = 0
                    self.path_list.pop()
        else:
            self.generation_complete = True
            self.path_finding_complete = True

class Button:
    """按钮类"""
    def __init__(self, x, y, width, height, text, color=GRAY, hover_color=DARK_GRAY, text_color=BLACK):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.text_color = text_color
        self.is_hovered = False
        self.is_active = False

    def draw(self, surface):
        """绘制按钮"""
        color = self.hover_color if self.is_hovered else self.color
        pygame.draw.rect(surface, color, self.rect)
        pygame.draw.rect(surface, BLACK, self.rect, 2)  # 边框
        
        text_surf = FONT_MEDIUM.render(self.text, True, self.text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

    def check_hover(self, pos):
        """检查鼠标是否悬停在按钮上"""
        self.is_hovered = self.rect.collidepoint(pos)
        return self.is_hovered

    def handle_event(self, event):
        """处理按钮事件"""
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.is_hovered:
                self.is_active = True
                return True
        return False

class Slider:
    """滑块控件"""
    def __init__(self, x, y, width, height, min_val, max_val, initial_val, text):
        self.rect = pygame.Rect(x, y, width, height)
        self.knob_rect = pygame.Rect(x, y, 20, height)
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial_val
        self.text = text
        self.dragging = False
        self.update_knob()

    def update_knob(self):
        """更新滑块位置"""
        ratio = (self.value - self.min_val) / (self.max_val - self.min_val)
        self.knob_rect.x = self.rect.x + int(ratio * (self.rect.width - self.knob_rect.width))

    def draw(self, surface):
        """绘制滑块"""
        # 绘制滑块轨道
        pygame.draw.rect(surface, GRAY, self.rect)
        pygame.draw.rect(surface, BLACK, self.rect, 2)
        
        # 绘制滑块
        pygame.draw.rect(surface, BLUE if self.dragging else DARK_GRAY, self.knob_rect)
        pygame.draw.rect(surface, BLACK, self.knob_rect, 2)
        
        # 绘制文本和值
        text_surf = FONT_SMALL.render(f"{self.text}: {self.value}", True, BLACK)
        text_rect = text_surf.get_rect(midleft=(self.rect.x, self.rect.y - 15))
        surface.blit(text_surf, text_rect)

    def handle_event(self, event):
        """处理滑块事件"""
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.knob_rect.collidepoint(event.pos):
                self.dragging = True
                return True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            # 计算新值
            relative_x = event.pos[0] - self.rect.x
            ratio = max(0, min(1, relative_x / self.rect.width))
            self.value = int(self.min_val + ratio * (self.max_val - self.min_val))
            self.update_knob()
            return True
        return False

class MazeGUI:
    """迷宫GUI主类"""
    def __init__(self):
        # 窗口设置
        self.width, self.height = 1000, 700
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        pygame.display.set_caption("迷宫生成与路径可视化工具")
        
        # 默认迷宫参数
        self.rows = 30
        self.cols = 30
        
        # 创建初始迷宫对象
        self.maze = None
        self.create_maze()
        
        # 控制变量
        self.running = True
        self.paused = False
        self.show_contour = True
        self.show_topology = False
        self.show_path = True
        self.show_centerline = False
        self.recording_path = False  # 是否正在记录路径生成过程
        self.frames = []  # 存储路径生成的帧
        self.topology_generated = False
        self.contour_saved = False
        self.video_saved = False
        
        # 创建UI控件
        self.create_ui_elements()
        
        # 迷宫绘制区域
        self.maze_rect = pygame.Rect(300, 50, 650, 600)
        
        # 输出文本
        self.output_text = ""
        self.path_coords_text = ""  # 存储路径坐标文本
        
    def create_ui_elements(self):
        """创建所有UI控件"""
        # 滑块
        self.row_slider = Slider(50, 50, 200, 20, 5, 50, self.rows, "迷宫行数")
        self.col_slider = Slider(50, 100, 200, 20, 5, 50, self.cols, "迷宫列数")
        
        # 按钮
        self.generate_btn = Button(50, 150, 200, 40, "生成迷宫", BLUE, (0, 0, 150))
        self.pause_btn = Button(50, 200, 200, 40, "暂停/继续", GREEN, (0, 150, 0))
        self.contour_btn = Button(50, 250, 200, 40, "显示/隐藏轮廓", ORANGE, (200, 100, 0))
        self.topology_btn = Button(50, 300, 200, 40, "显示/隐藏拓扑图", PINK, (100, 0, 100))
        self.centerline_btn = Button(50, 350, 200, 40, "显示/隐藏中心线", YELLOW, (200, 200, 0))
        self.path_btn = Button(50, 400, 200, 40, "显示/隐藏路径", RED, (150, 0, 0))
        
        # 保存按钮
        self.save_path_btn = Button(50, 450, 200, 40, "保存路径坐标", GRAY, DARK_GRAY)
        self.save_contour_btn = Button(50, 500, 200, 40, "保存轮廓图", GRAY, DARK_GRAY)
        self.save_topology_btn = Button(50, 550, 200, 40, "保存拓扑图", GRAY, DARK_GRAY)
        self.save_video_btn = Button(50, 600, 200, 40, "保存导航视频", GRAY, DARK_GRAY)
        
    def create_maze(self):
        """创建新的迷宫对象"""
        self.maze = Maze(self.rows, self.cols)
        self.topology_generated = False
        self.contour_saved = False
        self.video_saved = False
        self.frames = []
        self.output_text = ""
        self.recording_path = False  # 重置录制状态
        
    def handle_events(self):
        """处理所有事件"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            # 窗口大小调整
            elif event.type == pygame.VIDEORESIZE:
                self.width, self.height = event.w, event.h
                self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
                self.maze_rect = pygame.Rect(300, 50, self.width - 350, self.height - 100)
            
            # 处理滑块事件
            self.row_slider.handle_event(event)
            self.col_slider.handle_event(event)
            
            # 处理按钮事件
            if self.generate_btn.handle_event(event):
                self.rows = self.row_slider.value
                self.cols = self.col_slider.value
                self.create_maze()
                self.paused = False
                self.recording_path = True  # 开始记录路径生成过程
                
            if self.pause_btn.handle_event(event):
                self.paused = not self.paused
                
            if self.contour_btn.handle_event(event):
                self.show_contour = not self.show_contour
                
            if self.topology_btn.handle_event(event):
                self.show_topology = not self.show_topology
                
            if self.centerline_btn.handle_event(event):
                self.show_centerline = not self.show_centerline
                
            if self.path_btn.handle_event(event):
                self.show_path = not self.show_path
                
            if self.save_path_btn.handle_event(event) and self.maze.generation_complete:
                self.save_path_coordinates()
            
            if self.save_contour_btn.handle_event(event) and self.maze.generation_complete:
                self.save_contour_image()

            if self.save_topology_btn.handle_event(event) and self.maze.generation_complete:
                self.save_topology_image()
                
            if self.save_video_btn.handle_event(event) and self.maze.generation_complete and self.frames:
                self.save_navigation_video()
                
            # 检查鼠标悬停
            mouse_pos = pygame.mouse.get_pos()
            self.generate_btn.check_hover(mouse_pos)
            self.pause_btn.check_hover(mouse_pos)
            self.contour_btn.check_hover(mouse_pos)
            self.topology_btn.check_hover(mouse_pos)
            self.centerline_btn.check_hover(mouse_pos)
            self.path_btn.check_hover(mouse_pos)
            self.save_path_btn.check_hover(mouse_pos)
            self.save_contour_btn.check_hover(mouse_pos)
            self.save_topology_btn.check_hover(mouse_pos)
            self.save_video_btn.check_hover(mouse_pos)
    
    def show_path_coordinates(self):
        """在界面上显示路径坐标序列"""
        if not self.maze.generation_complete or not self.maze.path_list:
            self.path_coords_text = "迷宫尚未生成完成或没有路径数据"
            return
        
        # 只显示部分坐标以避免界面混乱
        if len(self.maze.path_list) <= 10:
            coords_str = " -> ".join(f"({x},{y})" for x, y in self.maze.path_list)
        else:
            first_part = " -> ".join(f"({x},{y})" for x, y in self.maze.path_list[:3])
            last_part = " -> ".join(f"({x},{y})" for x, y in self.maze.path_list[-3:])
            coords_str = f"{first_part} -> ... -> {last_part} (共{len(self.maze.path_list)}个点)"
        
        self.path_coords_text = f"路径坐标: {coords_str}"

    def save_path_coordinates(self):
        """保存路径坐标到CSV文件"""
        if not self.maze.generation_complete or not self.maze.path_list:
            self.output_text = "迷宫尚未生成完成或没有路径数据"
            return
            
        try:
            with open('路径坐标序列.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['X', 'Y'])  # 写入标题行
                for point in self.maze.path_list:
                    writer.writerow(point)
            self.output_text = "路径坐标已保存为 路径坐标序列.csv"
        except Exception as e:
            self.output_text = f"保存失败: {str(e)}"
    
    def save_topology_image(self):
        """保存拓扑结构图为图片"""
        if not self.maze.generation_complete:
            self.output_text = "迷宫尚未生成完成"
            return
            
        try:
            '''
            # 计算中心线
            centerline = self.compute_centerline()
            
            # 创建图形
            graph = nx.Graph()
            
            # 添加节点和边
            for i, node in enumerate(centerline):
                graph.add_node(node)
                if i > 0:  # 将相邻的中心点连接起来
                    graph.add_edge(centerline[i - 1], node)
            
            # 设置节点位置并绘制图形
            pos = {node: (node[1], -node[0]) for node in graph.nodes()}
            plt.figure(figsize=(10, 10))
            nx.draw(graph, pos, with_labels=True, 
                    node_color="orange", edge_color="blue", 
                    node_size=100, font_size=10)
            plt.title("Maze Path Topology")
            '''
            # 计算单元格大小以适应绘制区域
            cell_width = self.maze_rect.width / self.maze.COLUMNS
            cell_height = self.maze_rect.height / self.maze.ROWS
            cell_size = min(cell_width, cell_height)
            # 使用 DFS 查找路径
            start = (self.maze.start[0], self.maze.start[1])
            end = (self.maze.ROWS - 1, self.maze.COLUMNS - 1)
            dfs_path = self.dfs_path(start, end)

            # 创建图形
            graph = nx.Graph()
            for i, node in enumerate(dfs_path):
                graph.add_node(node)
                if i > 0:
                    graph.add_edge(dfs_path[i - 1], node)

            # 设置节点位置并绘制图形
            pos = {node: (node[1], -node[0]) for node in graph.nodes()}
            plt.figure(figsize=(10, 10))
            nx.draw(graph, pos, with_labels=True,
                    node_color="orange", edge_color="blue",
                    node_size=100, font_size=10)
            plt.title("Maze Path Topology")
            # 保存图像
            plt.savefig("拓扑图.png")
            plt.close()
            self.output_text = "拓扑结构图已保存为 拓扑图.png"
        except Exception as e:
            self.output_text = f"保存拓扑图失败: {str(e)}"
    
    def save_contour_image(self):
        """保存迷宫轮廓图为图片"""
        if not self.maze.generation_complete:
            self.output_text = "迷宫尚未生成完成"
            return
            
        try:
            # 创建一个临时Surface来绘制轮廓图
            cell_size = 20  # 固定单元格大小以确保清晰度
            width = self.maze.COLUMNS * cell_size
            height = self.maze.ROWS * cell_size
            surface = pygame.Surface((width, height))
            surface.fill(WHITE)
            
            # 绘制迷宫轮廓
            contours = self.extract_contours()
            for (start, end) in contours:
                x1, y1 = start
                x2, y2 = end
                pygame.draw.line(
                    surface, BLACK, 
                    (y1 * cell_size, x1 * cell_size), 
                    (y2 * cell_size, x2 * cell_size), 
                    2  # 线条宽度
                )
            
            # 保存图像
            pygame.image.save(surface, "路径轮廓图.png")
            self.output_text = "迷宫轮廓图已保存为 路径轮廓图.png"
        except Exception as e:
            self.output_text = f"保存轮廓图失败: {str(e)}"
            
    def save_navigation_video(self):
        """保存导航演示视频"""
        if not self.frames:
            self.output_text = "没有可保存的视频帧"
            return
            
        try:
            import imageio
            # 确保frames目录存在
            if not os.path.exists('frames'):
                os.makedirs('frames')
            
            # 保存每一帧为图片
            for i, frame in enumerate(self.frames):
                pygame.image.save(frame, f'frames/frame_{i:04d}.png')
            
            # 使用imageio创建视频
            images = []
            for i in range(len(self.frames)):
                images.append(imageio.imread(f'frames/frame_{i:04d}.png'))
            
            imageio.mimsave('导航演示视频.mp4', images, fps=30)
            
            # 清理临时文件
            for i in range(len(self.frames)):
                os.remove(f'frames/frame_{i:04d}.png')
            os.rmdir('frames')
            
            self.output_text = "导航演示视频已保存为 导航演示视频.mp4"
            self.video_saved = True
        except Exception as e:
            self.output_text = f"保存视频失败: {str(e)}"
    
    def compute_centerline(self):
        """计算迷宫通道的中心线点"""
        centerline = []
        # 处理水平方向的通道
        for i in range(self.maze.ROWS):
            j = 0
            while j < self.maze.COLUMNS:
                if self.maze.matrix[i][j] in (0, 2, 3):
                    start = j
                    while j < self.maze.COLUMNS and self.maze.matrix[i][j] in (0, 2, 3):
                        j += 1
                    end = j - 1
                    mid = (start + end) // 2
                    centerline.append((i, mid))
                j += 1
        # 处理垂直方向的通道
        for j in range(self.maze.COLUMNS):
            i = 0
            while i < self.maze.ROWS:
                if self.maze.matrix[i][j] in (0, 2, 3):
                    start = i
                    while i < self.maze.ROWS and self.maze.matrix[i][j] in (0, 2, 3):
                        i += 1
                    end = i - 1
                    mid = (start + end) // 2
                    pt = (mid, j)
                    if pt not in centerline:
                        centerline.append(pt)
                i += 1
        return centerline
    
    def extract_contours(self):
        """提取迷宫轮廓线段"""
        contours = []
        for i in range(self.maze.ROWS):
            for j in range(self.maze.COLUMNS):
                # 对于每个通道格子（包括路径和终点），检查其四周是否需要绘制边界
                if self.maze.matrix[i][j] in (0, 2, 3):
                    # 检查上边界
                    if i == 0 or self.maze.matrix[i - 1][j] == 1:
                        contours.append(((i, j), (i, j + 1)))
                    # 检查下边界
                    if i == self.maze.ROWS - 1 or self.maze.matrix[i + 1][j] == 1:
                        contours.append(((i + 1, j), (i + 1, j + 1)))
                    # 检查左边界
                    if j == 0 or self.maze.matrix[i][j - 1] == 1:
                        contours.append(((i, j), (i + 1, j)))
                    # 检查右边界
                    if j == self.maze.COLUMNS - 1 or self.maze.matrix[i][j + 1] == 1:
                        contours.append(((i, j + 1), (i + 1, j + 1)))
        return contours
    
    def draw_contours(self, cell_size):
        """绘制迷宫轮廓线"""
        contours = self.extract_contours()
        for (start, end) in contours:
            x1, y1 = start
            x2, y2 = end
            pygame.draw.line(
                self.screen, BLUE, 
                (self.maze_rect.x + y1 * cell_size, self.maze_rect.y + x1 * cell_size), 
                (self.maze_rect.x + y2 * cell_size, self.maze_rect.y + x2 * cell_size), 
                2  # 线条宽度
            )
    def dfs_path(self, start, end):
        """使用深度优先搜索查找从起点到终点的路径"""
        stack = [start]
        visited = set()
        path = []

        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            path.append(current)

            # 如果到达终点，返回路径
            # if current == end:
                # return path
            
            # 获取当前点的邻居
            neighbors = []
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # 上下左右
                nx, ny = current[0] + dx, current[1] + dy
                if 0 <= nx < self.maze.ROWS and 0 <= ny < self.maze.COLUMNS:
                    if self.maze.matrix[nx][ny] in (0, 2, 3) and (nx, ny) not in visited:
                        neighbors.append((nx, ny))

            # 将邻居加入栈
            stack.extend(neighbors)

        return path  

    def draw_topology_with_dfs(self, cell_size):
        """基于 DFS 绘制拓扑图"""
        if not self.maze.generation_complete:
            return

        # 使用 DFS 查找所有路径
        start = (self.maze.start[0], self.maze.start[1])
        end = (self.maze.ROWS - 1, self.maze.COLUMNS - 1)
        dfs_path = self.dfs_path(start, end)

        # 绘制路径点（节点）
        for i, node in enumerate(dfs_path):
            x1 = self.maze_rect.x + node[1] * cell_size + cell_size // 2
            y1 = self.maze_rect.y + node[0] * cell_size + cell_size // 2
            pygame.draw.circle(self.screen, PINK, (x1, y1), cell_size // 4)

            # 绘制路径边（相邻点连接）
            if i > 0:
                prev_node = dfs_path[i - 1]
                x2 = self.maze_rect.x + prev_node[1] * cell_size + cell_size // 2
                y2 = self.maze_rect.y + prev_node[0] * cell_size + cell_size // 2
                pygame.draw.line(self.screen, BLUE, (x1, y1), (x2, y2), 2)

    def draw_maze(self):
        """绘制迷宫"""
        if not self.maze:
            return
            
        # 计算单元格大小以适应绘制区域
        cell_width = self.maze_rect.width / self.maze.COLUMNS
        cell_height = self.maze_rect.height / self.maze.ROWS
        cell_size = min(cell_width, cell_height)
        
        # 绘制背景
        pygame.draw.rect(self.screen, WHITE, self.maze_rect)
        
        # 绘制迷宫单元格
        for i in range(self.maze.ROWS):
            for j in range(self.maze.COLUMNS):
                x = self.maze_rect.x + j * cell_size
                y = self.maze_rect.y + i * cell_size
                rect = pygame.Rect(x, y, cell_size, cell_size)
                
                if self.maze.matrix[i][j] == 1:  # 墙
                    pygame.draw.rect(self.screen, GRAY, rect)
                elif self.maze.matrix[i][j] == 2 and self.show_path:  # 路径
                    pygame.draw.rect(self.screen, GREEN, rect)
                elif self.maze.matrix[i][j] == 3:  # 终点
                    pygame.draw.rect(self.screen, RED, rect)
                elif self.maze.matrix[i][j] == 0:  # 通道
                    pygame.draw.rect(self.screen, WHITE, rect)
        
        # 绘制起点
        start_x = self.maze_rect.x + self.maze.start[1] * cell_size
        start_y = self.maze_rect.y + self.maze.start[0] * cell_size
        pygame.draw.rect(self.screen, BLUE, (start_x, start_y, cell_size, cell_size))
        
        # 绘制轮廓线
        if self.show_contour:
            self.draw_contours(cell_size)
        
        # 绘制中心线点
        if self.show_centerline and self.maze.generation_complete:
            centerline = self.compute_centerline()
            for (i, j) in centerline:
                center_x = self.maze_rect.x + j * cell_size + cell_size // 2
                center_y = self.maze_rect.y + i * cell_size + cell_size // 2
                pygame.draw.circle(self.screen, ORANGE, (center_x, center_y), cell_size // 6)
        
        # 绘制拓扑图
        if self.show_topology and self.maze.generation_complete:
            self.draw_topology_with_dfs(cell_size)
        
        # 绘制边框
        # pygame.draw.rect(self.screen, BLACK, self.maze_rect, 2)
        
        # 如果正在记录路径生成过程，保存当前帧
        if self.recording_path and not self.maze.generation_complete:
            frame = pygame.Surface((self.maze_rect.width, self.maze_rect.height))
            frame.blit(self.screen, (0, 0), self.maze_rect)
            self.frames.append(frame)
            
            # 如果迷宫生成完成，停止录制
            if self.maze.generation_complete:
                self.recording_path = False
                self.output_text = "路径生成过程已记录，点击保存视频按钮保存为视频"
    
    def draw_ui(self):
        """绘制用户界面"""
        # 绘制背景
        self.screen.fill(WHITE)
        
        # 绘制滑块
        self.row_slider.draw(self.screen)
        self.col_slider.draw(self.screen)
        
        # 绘制按钮
        self.generate_btn.draw(self.screen)
        self.pause_btn.draw(self.screen)
        self.contour_btn.draw(self.screen)
        self.topology_btn.draw(self.screen)
        self.centerline_btn.draw(self.screen)
        self.path_btn.draw(self.screen)
        self.save_path_btn.draw(self.screen)
        self.save_contour_btn.draw(self.screen)
        self.save_topology_btn.draw(self.screen)
        self.save_video_btn.draw(self.screen)
        
        # 绘制状态信息
        status_text = f"状态: {'生成中' if not self.maze.generation_complete else '已完成'}"
        if self.paused:
            status_text += " (已暂停)"
        status_surf = FONT_SMALL.render(status_text, True, BLACK)
        self.screen.blit(status_surf, (50, 650))
        
        # 绘制输出信息
        if self.output_text:
            output_surf = FONT_SMALL.render(self.output_text, True, BLACK)
            self.screen.blit(output_surf, (300, 680))

        # 显示路径坐标
        if self.maze.generation_complete:
            self.show_path_coordinates()
            path_surf = FONT_SMALL.render(self.path_coords_text, True, BLACK)
            self.screen.blit(path_surf, (300, 660))
        
        # 绘制迷宫
        self.draw_maze()
    
    def update(self):
        """更新游戏状态"""
        if not self.paused and self.maze and not self.maze.generation_complete:
            self.maze.matrix_generate()
    
    def run(self):
        """主循环"""
        clock = pygame.time.Clock()
        while self.running:
            self.handle_events()
            self.update()
            self.draw_ui()
            pygame.display.flip()
            clock.tick(60)
        
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    app = MazeGUI()
    app.run()