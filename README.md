## pybaseutils(Python)
pybaseutils是个人开发常用的python库，集成了python等常用的算法
- 安装方法1：pip install pybaseutils -i https://pypi.tuna.tsinghua.edu.cn/simple (有延时，可能不是最新版本)
- 安装方法2：pip install --upgrade pybaseutils -i https://pypi.org/simple (从pypi源下载最新版本)

## 目录结构

```
├── base_utils         # base_utils的C++源代码
├── pybaseutils        # pybaseutils的python源代码
├── data               # 相关测试数据
├── test               # base_utils的测试代码
│   ├── build.sh
│   ├── CMakeLists.txt
│   ├── kalman_test.cpp
│   └── main.cpp
└── README.md

```

## python 测试 Demo 
..\pybaseutils\cvutil\mouse_utils.py
