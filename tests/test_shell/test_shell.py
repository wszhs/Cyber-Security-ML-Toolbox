'''
Author: your name
Date: 2021-04-20 15:03:44
LastEditTime: 2021-04-20 15:33:07
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/tests/test_shell/test_shell.py
'''
import os


# 遍历文件夹
def walkFile(file):
    for root, dirs, files in os.walk(file):

        # root 表示当前正在访问的文件夹路径
        # dirs 表示该文件夹下的子目录名list
        # files 表示该文件夹下的文件list

        # 遍历文件
        # for f in files:
        #     print(os.path.join(root, f))

        # # 遍历所有的文件夹
        for d in dirs:
            print(os.path.join(root, d))
            os.system('/Applications/Wireshark.app/Contents/MacOS/mergecap -w dowgin.pcap ./Dowgin/*.pcap')


def main():
    walkFile("/Volumes/data/数据集/cicandroid/pcap/Adware")


if __name__ == '__main__':
    main()
