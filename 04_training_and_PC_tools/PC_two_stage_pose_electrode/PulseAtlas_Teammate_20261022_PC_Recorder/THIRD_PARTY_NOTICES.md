# 组件来源与版本

本包为用户提供Fusion源码和原权重的适配及微调派生版本，Ultralytics相关AGPL-3.0许可文本保留于licenses/。本轮第二阶段权重已经微调，不再是原始冻结权重；第一阶段ONNX、标定和位置匹配源码按原文件保留。

本轮已验证训练/推理环境是Ultralytics8.3.163、PyTorch2.5.0+cu118，依赖由environment.yml和requirements-conda.lock.txt声明；依赖安装包/完整框架源码不包含在ZIP中。原Fusion包使用8.4.80，隔离兼容抽查记录在reports/runtime_compatibility.json。原包的来源声明另存docs/SOURCE_THIRD_PARTY_NOTICES.md，仅作为历史来源，不代表本包依赖版本。

项目训练/评估实现放在training_code/供复核；重训仍需原始来源模型包及当前训练数据，不包含真人原图。本包不扩大既有版权或授权范围。
