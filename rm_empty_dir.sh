#!/bin/bash
# 查找 checkpoints 目录下所有的空文件夹并删除
find checkpoints/ -type d -empty -delete
echo "已清除 checkpoints/ 下的空文件夹"
