## python 命令行工具  argparse基本用法

#### 基本框架

  直接上代码

```python
import argparse

def main():
    parser = argparse.ArgumentParser(description="Demo of argparse")
    # 上面这句话生成一个parser对象，括号里为对象描述，用来干嘛的
    parser.add_argument('-n','--name', default=' Li ')
    # 添加一个参数，'-n'和‘-name’指向同一个参数，default为参数的默认值，下同
    # 这里要注意个问题，当'-'和'--'同时出现的时候，系统默认后者为参数名，前者不是，但是在命令行输入的时候没有这个区分接下来就是打印参数信息了。
    parser.add_argument('-y','--year', default='20')
    args = parser.parse_args()
    # parser.parse_args()用于解析获得的参数
    # 如果全为默认参数，则此处会输出
    # <<<Namespace(name="Li",year="20")
    # 相当于一个对象，有name和year两个属性
    print(args)
    name = args.name
    year = args.year
    print('Hello {}  {}'.format(name,year))

if __name__ == '__main__':
    main()
```

#### 测试如下

```python
python parser_demo.py
<<< Namespace(name=' Li ', year='20')
<<< Hello  Li   20
# 注意，2处默认两个默认参数值，Li和20

python parser_demo.py --name Wang -y 26  #使用-n 或者 --name。-y或者--year
<<< Namespace(name='Wang', year='21')
<<< Hello Wang  21
# 此处不是默认参数了

```

#### help提示：

```python
python parser_demo.py -h
<<< 
usage: parser_demo.py [-h] [-n NAME] [-y YEAR]

Demo of argparse

optional arguments:
  -h, --help            show this help message and exit
  -n NAME, --name NAME
  -y YEAR, --year YEAR
```

#### add_argument全部参数

```python
ArgumentParser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
'''
定义应该如何解析一个命令行参数。下面每个参数有它们自己详细的描述，简单地讲它们是：
name or flags - 选项字符串的名字或者列表，例如foo 或者-f, --foo。
action - 在命令行遇到该参数时采取的基本动作类型。
nargs - 应该读取的命令行参数数目。
const - 某些action和nargs选项要求的常数值。
default - 如果命令行中没有出现该参数时的默认值。
type - 命令行参数应该被转换成的类型。
choices - 参数可允许的值的一个容器。
required - 该命令行选项是否可以省略（只针对可选参数）。
help - 参数的简短描述。
metavar - 参数在帮助信息中的名字。
dest - 给parse_args()返回的对象要添加的属性名称。
'''
```

#### mmdetection 中 coco_eval.py

```python
from argparse import ArgumentParser

from mmdet.core import coco_eval

def main():
    parser = ArgumentParser(description='COCO Evaluation')
    parser.add_argument('result', help='result file path')
    parser.add_argument('--ann', help='annotation file path')
    parser.add_argument(
        '--types',
        type=str,
        nargs='+',
        choices=['proposal_fast', 'proposal', 'bbox', 'segm', 'keypoint'],
        default=['bbox'],
        help='result types')
    # nargs = '+'。和'*'一样，出现的所有命令行参数都被收集到一个列表中。除此之外，如果没有至少出现一个命令行参数将会产生一个错误信息。例如：
    parser.add_argument(
        '--max-dets',
        type=int,
        nargs='+',
        default=[100, 300, 1000],
        help='proposal numbers, only used for recall evaluation')
    args = parser.parse_args()
    coco_eval(args.result, args.types, args.ann, args.max_dets)

if __name__ == '__main__':
    main()
```



