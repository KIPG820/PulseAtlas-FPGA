# 组件来源和许可

Ultralytics YOLO代码、官方预训练及派生权重来自Ultralytics。本包提供同版本框架源码、项目适配代码、微调权重、训练配置与评估脚本；相关AGPL-3.0许可文本在licenses目录。源码版本8.4.80。许可及再分发说明：https://docs.ultralytics.com/help/contributing ，https://www.ultralytics.com/license 。本包没有授予闭源商业豁免。

人体姿态ONNX、解码器及原标定JSON从用户提供的PulseAtlas原始交接包复制。模型清单存在一个63字符摘要录入错误，实际ONNX按交接包顶层SHA256SUMS及复制校验核实。原适配器源码保持原样，包装/同帧融合/几何匹配/便携路径修改见app源文件；这些修改不改变两模型权重。

Python、PyTorch、torchvision、OpenCV、NumPy、Pillow等通过安装脚本获取，遵循各自许可；安装的依赖不包含在ZIP里。Python：https://www.python.org/ ，PyTorch：https://pytorch.org/ ，OpenCV：https://opencv.org/ 。既有材料的版权和授权保留，不因打包而扩大。
