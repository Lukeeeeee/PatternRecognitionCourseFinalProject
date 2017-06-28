# PatternRecognitionCourseFinalProject
### 1. 需预先安装的包和工具:

其中2-5推荐使用 pip包管理器 安装

```shell
1. python 2.7
2. tensorflow >= 1.1.0
3. Pillow >= 4.1.1
4. docopt >= 0.6.2
5. numpy >= 1.11.2
```

### 2. 如何使用代码
入口程序为:
```shell
PatternRecognitionCourseFinalProject\test\test.py

Usage:
    test.py train
    test.py test model_id <model_id>

```
如果是训练：
```shell
cd PatternRecognitionCourseFinalProject\test
python test.py train
```
如果是测试：
```shell
cd PatternRecognitionCourseFinalProject\test
python test.py test model_id /demo/6-14-19-4-12/model/model.ckpt-500 # 主目录下某个模型文件
```

 