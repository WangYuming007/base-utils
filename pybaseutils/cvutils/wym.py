import cv2
import numpy as np
from typing import Callable
from pybaseutils import image_utils

class DrawImageMouse(object):
    """使用鼠标绘图"""

    def __init__(self, max_point=-1, line_color=(0, 0, 255), text_color=(255, 0, 0), thickness=2):
        """
        :param max_point: 最多绘图的点数，超过后将绘制无效；默认-1表示无限制
        :param line_color: 线条的颜色
        :param text_color: 文本的颜色
        :param thickness: 线条粗细
        """
        # 初始化鼠标绘图的一些参数
        self.max_point = max_point
        self.line_color = line_color
        self.text_color = text_color
        self.text_size = max(int(thickness * 0.4), 0.5)
        self.focus_color = (0, 255, 0)  # 鼠标焦点的颜色
        self.focus_size = max(int(thickness * 2), 6)  # 鼠标焦点的颜色
        self.thickness = thickness
        self.key = -1  # 键盘值
        self.orig = None  # 原始图像
        self.last = None  # 上一帧
        self.next = None  # 下一帧或当前帧
        self.polygons = np.zeros(shape=(0, 2), dtype=np.int32)  # 鼠标绘制点集合
        self.lineset = []  # 鼠标绘制的直线集合

    def clear(self):
        # 清空鼠标绘制的数据
        self.key = -1
        #清空 self.polygons，这是存储鼠标绘制点的数组，以便重新开始新的绘制
        self.polygons = np.zeros(shape=(0, 2), dtype=np.int32)
        self.lines = []  # 清空直线集合
        if self.orig is not None: self.last = self.orig.copy()
        if self.orig is not None: self.next = self.orig.copy()

    def get_lineset(self):
        """获得直线集合数据"""
        return self.lineset

    def task(self, image, callback: Callable, winname="winname"):
        """
        鼠标监听任务
        :param image: 图像
        :param callback: 鼠标回调函数
        :param winname: 窗口名称
        :return:
        """
        self.orig = image.copy()
        self.last = image.copy()
        self.next = image.copy()
        cv2.namedWindow(winname, flags=cv2.WINDOW_NORMAL)
        #将窗口winname的鼠标事件绑定到event_draw_rectangle和event_draw_polygon这两个回调函数。当用户在该窗口进行鼠标操作时，相应的回调函数会被调用。
        cv2.setMouseCallback(winname, callback, param={"winname": winname})
        while True:
            self.key = self.show_image(winname, self.next, delay=25)
            if (self.key == 13 or self.key == 32) and len(self.polygons) > 0:  # 按空格32和回车键13表示完成绘制
                break
            elif self.key == 27:  # 按ESC退出程序
                exit(0)
            elif self.key == 99:  # 按键盘c重新绘制
                self.clear()

    def event_default(self, event, x, y, flags, param):
        pass

    def event_draw_lineset(self, event, x, y, flags, param):
        exceed = self.max_point > 0 and len(self.lineset) >= self.max_point
        self.next = self.last.copy()
        point = (x, y)
        text = str(len(self.lineset))
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(self.next, point, radius=self.focus_size, color=self.focus_color, thickness=self.thickness)
            cv2.putText(self.next, text, point, cv2.FONT_HERSHEY_SIMPLEX, self.text_size, self.text_color,
                        self.thickness)
            if len(self.lineset) > 0 and len(self.lineset) % 2 == 1:
                cv2.line(self.next, self.lineset[-1], point, color=self.line_color, thickness=self.thickness)
            if not exceed:
                self.last = self.next
                self.lineset.append(point)
        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):# 按住左键拖曳
            cv2.circle(self.next, point, radius=self.focus_size, color=self.focus_color, thickness=self.thickness)
            if len(self.lineset) > 0 and len(self.lineset) % 2 == 1:
                cv2.line(self.next, self.lineset[-1], point, color=self.line_color, thickness=self.thickness)

    @staticmethod
    def show_image(self, title, image, delay=5):
        """显示图像"""
        cv2.imshow(title, image)
        key = cv2.waitKey(delay=delay) if delay >= 0 else -1
        return key

    def draw_image_lineset_on_mouse(self, image, winname="draw_lineset"):
        self.task(image, callback=self.event_draw_lineset, winname=winname)
        lineset = self.get_lineset()
        return lineset

def draw_image_lineset_on_mouse_example(image_file, winname="draw_lineset"):
    """
    获得鼠标绘制的直线集合
    :param image_file:
    :param winname: 窗口名称
    :return: lineset is [(x1, y1, x2, y2), ...]
    """
    image = cv2.imread(image_file)
    mouse = DrawImageMouse()  # 创建DrawImageMouse对象
    lineset = mouse.draw_image_lineset_on_mouse(image, winname=winname)  # 调用DrawImageMouse对象的draw_image_lineset_on_mouse方法
    image = image_utils.draw_image_lines(image, lineset, thickness=2)  # 在原始图像上绘制直线集合
    mouse.show_image(winname, image, delay=0)
    return lineset

if __name__ == '__main__':
    image_file = "../../data/test.png"
    out =  draw_image_lineset_on_mouse_example(image_file)
    print(out)
