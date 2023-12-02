# -*-coding: utf-8 -*-
"""
    @Author : Pan
    @E-mail : 390737991@qq.com
    @Date   : 2022-07-27 15:23:24
    @Brief  :
"""

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
        self.lineset = []  # 清空直线集合
        if self.orig is not None: self.last = self.orig.copy()
        if self.orig is not None: self.next = self.orig.copy()

    def get_polygons(self):
        """获得多边形数据"""
        return self.polygons

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
            if (self.key == 13 or self.key == 32) and len(self.lineset) > 0:  # 按空格32和回车键13表示完成绘制
                print("there's an enter!")
                break
            elif self.key == 27:  # 按ESC退出程序
                exit(0)
            elif self.key == 99:  # 按键盘c重新绘制
                print("there's a c!")
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
            #cv2.putText(self.next, text, point, cv2.FONT_HERSHEY_SIMPLEX, self.text_size, self.text_color,self.thickness)
            if len(self.lineset) > 0 and len(self.lineset) % 2 == 1:
                cv2.line(self.next, self.lineset[-1], point, color=self.line_color, thickness=self.thickness)
            if not exceed:
                self.last = self.next
                self.lineset.append(point)
        #elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):# 按住左键拖曳
        else:
            cv2.circle(self.next, point, radius=self.focus_size, color=self.focus_color, thickness=self.thickness)
            if len(self.lineset) > 0 and len(self.lineset) % 2 == 1:
                cv2.line(self.next, self.lineset[-1], point, color=self.line_color, thickness=self.thickness)
            
    def event_draw_rectangle(self, event, x, y, flags, param):
        """绘制矩形框"""
        if len(self.polygons) == 0: self.polygons = np.zeros(shape=(2, 2), dtype=np.int32)  # 多边形轮廓
        point = (x, y)# 获取当前鼠标位置

        # 处理鼠标事件
        if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击,则在原图打点
            self.next = self.last.copy()# 复制上一帧图像，以便在上面进行绘制
            self.polygons[0, :] = point# 记录鼠标点击的位置作为多边形的第一个点
            # 在图像上绘制鼠标点击位置的圆圈
            cv2.circle(self.next, point, radius=self.focus_size, color=self.focus_color, thickness=self.thickness)
        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):  # 按住左键拖曳，画框
            self.next = self.last.copy()# 复制上一帧图像，以便在上面进行绘制
            # 在图像上绘制鼠标点击位置和当前位置之间的矩形框
            cv2.circle(self.next, self.polygons[0, :], radius=self.focus_size, color=self.focus_color,
                       thickness=self.thickness)
            cv2.circle(self.next, point, radius=self.focus_size, color=self.focus_color, thickness=self.thickness)
            cv2.rectangle(self.next, self.polygons[0, :], point, color=self.line_color, thickness=self.thickness)
        elif event == cv2.EVENT_LBUTTONUP:  # 左键释放，显示
            self.next = self.last.copy()
            # 记录鼠标释放的位置作为多边形的第二个点
            self.polygons[1, :] = point
            # 在图像上绘制最终的矩形框
            cv2.rectangle(self.next, self.polygons[0, :], point, color=self.line_color, thickness=self.thickness)

    def event_draw_polygon(self, event, x, y, flags, param):
        """绘制多边形"""
        # 判断是否超过最大点数限制
        exceed = self.max_point > 0 and len(self.polygons) >= self.max_point
        self.next = self.last.copy() # 复制上一帧图像，以便在上面进行绘制
        point = (x, y) # 获取当前鼠标位置
        # 标注文本信息，显示多边形的顶点数
        text = str(len(self.polygons))
        if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击,则在原图打点
            # 在图像上绘制鼠标点击位置的圆圈
            cv2.circle(self.next, point, radius=self.focus_size, color=self.focus_color, thickness=self.thickness)
            # 在鼠标点击位置显示顶点数文本
            cv2.putText(self.next, text, point, cv2.FONT_HERSHEY_SIMPLEX, self.text_size, self.text_color,
                        self.thickness)
            # 如果已有点，则绘制当前点与上一个点之间的直线
            if len(self.polygons) > 0:
                cv2.line(self.next, self.polygons[-1, :], point, color=self.line_color, thickness=self.thickness)
            # 如果未超过点数限制，则将当前点添加到多边形的点集合中
            if not exceed:
                self.last = self.next
                self.polygons = np.concatenate([self.polygons, np.array(point).reshape(1, 2)])
        else:
            # 在图像上绘制鼠标当前位置的圆圈
            cv2.circle(self.next, point, radius=self.focus_size, color=self.focus_color, thickness=self.thickness)
            # 如果已有点，则绘制当前点与上一个点之间的直线
            if len(self.polygons) > 0:
                cv2.line(self.next, self.polygons[-1, :], point, color=self.line_color, thickness=self.thickness)


    @staticmethod
    def polygons2box(polygons):
        """将多边形转换为矩形框"""
        xmin = min(polygons[:, 0])
        ymin = min(polygons[:, 1])
        xmax = max(polygons[:, 0])
        ymax = max(polygons[:, 1])
        return [xmin, ymin, xmax, ymax]

    def show_image(self, title, image, delay=5):
        """显示图像"""
        cv2.imshow(title, image)
        key = cv2.waitKey(delay=delay) if delay >= 0 else -1
        return key

    def draw_image_lineset_on_mouse(self, image, winname="draw_lineset"):
        print("enter  task")
        self.task(image, callback=self.event_draw_lineset, winname=winname)
        print("leave task")
        lineset = self.get_lineset()
        return lineset

    def draw_image_rectangle_on_mouse(self, image, winname="draw_rectangle"):
        """
        获得鼠标绘制的矩形框box=[xmin,ymin,xmax,ymax]
        :param image:
        :param winname: 窗口名称
        :return: box is[xmin,ymin,xmax,ymax]
        """
        #设置callback函数
        self.task(image, callback=self.event_draw_rectangle, winname=winname)
        polygons = self.get_polygons()
        box = self.polygons2box(polygons)
        return box

    def draw_image_polygon_on_mouse(self, image, winname="draw_polygon"):
        """
        获得鼠标绘制的矩形框box=[xmin,ymin,xmax,ymax]
        :param image:
        :param winname: 窗口名称
        :return: polygons is (N,2)
        """
        self.task(image, callback=self.event_draw_polygon, winname=winname)
        polygons = self.get_polygons()
        return polygons

def draw_lineset(image, lineset, thickness=2):
    # 遍历直线集合中的每一条线
    for i in range(0, len(lineset), 2):
        # 获取偶数点和下一个点的坐标
        x0,y0 = lineset[i]
        x1,y1 = lineset[i + 1]
        cv2.circle(image, (x0, y0), radius=6, color=(0, 255, 0), thickness=thickness)
        cv2.circle(image, (x1, y1), radius=6, color=(0, 255, 0), thickness=thickness)
        cv2.line(image, (x0, y0), (x1, y1), color=(0, 0, 255), thickness=thickness)
    return image
    
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
    #image = image_utils.draw_image_lines(image, lineset, thickness=2)  # 在原始图像上绘制直线集合
    image = draw_lineset(image, lineset, thickness=2)
    mouse.show_image(winname, image, delay=0)
    return lineset


def draw_image_rectangle_on_mouse_example(image_file, winname="draw_rectangle"):
    """
    获得鼠标绘制的矩形框
    :param image_file:
    :param winname: 窗口名称
    :return: box=[xmin,ymin,xmax,ymax]
    """
    # 读取图像文件
    image = cv2.imread(image_file)
    # 创建DrawImageMouse对象
    mouse = DrawImageMouse()
    # 调用DrawImageMouse对象的draw_image_rectangle_on_mouse方法，该方法将进入事件监听循环，直到满足退出条件
    box = mouse.draw_image_rectangle_on_mouse(image, winname=winname)
    # 根据绘制的矩形框的坐标截取原始图像的感兴趣区域（ROI）
    roi: np.ndarray = image[box[1]:box[3], box[0]:box[2]]
    # 如果ROI非空，则显示ROI图像
    if roi.size > 0: mouse.show_image("Image ROI", roi)
    # 在原始图像上绘制最终的矩形框
    image = image_utils.draw_image_boxes(image, [box], color=(0, 0, 255), thickness=2)
    # 显示带有矩形框的原始图像
    mouse.show_image(winname, image, delay=0)
    # 返回绘制的矩形框的坐标
    return box

def draw_image_polygon_on_mouse_example(image_file, winname="draw_polygon"):
    """
    获得鼠标绘制的多边形
    :param image_file:
    :param winname: 窗口名称
    :return: polygons is (N,2)
    """
    image = cv2.imread(image_file)
    # 通过鼠标绘制多边形
    mouse = DrawImageMouse(max_point=-1)
    polygons = mouse.draw_image_polygon_on_mouse(image, winname=winname)
    image = image_utils.draw_image_points_lines(image, polygons, thickness=2)
    mouse.show_image(winname, image, delay=0)
    return polygons

if __name__ == '__main__':
    image_file = "../../data/road.png"
    # 绘制矩形框
    #out = draw_image_rectangle_on_mouse_example(image_file)
    # 绘制多边形
    #out = draw_image_polygon_on_mouse_example(image_file)
    # 绘制直线集合
    out =  draw_image_lineset_on_mouse_example(image_file)
    print(out)
