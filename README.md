# 深度学习环境-base

## 项目构建流程

[【2025深度学习环境搭建-2】pytorch+Docker+VS Code+DevContainer搭建本地深度学习环境](https://blog.csdn.net/m0_63070489/article/details/145813739)

## pip环境导入导出

从requirements.txt导入环境：

`pip install --no-cache-dir -r requirements.txt`

导出环境到文件requirements.txt：

`pip freeze | grep -v '@ file://' > requirements.txt`