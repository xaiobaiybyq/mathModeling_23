# 问题一 Git 交付说明

本目录在原有 question1_submission.zip 的基础上补齐完整论文与生成脚本。可直接阅读[15页完整 Word 报告](问题一_分析与数学建模_完整版.docx)或[PDF 预览](paper_work/render/paper.pdf)；100条自主提取特征位于 results/features/，逐条元数据及运行记录位于 results/。

已纳入 Git 的材料包括：100条词级三模态特征NPZ、结果CSV与核验记录、完整Word报告和PDF预览、六张论文制图、最终报告生成源码 paper_work/build_paper.py、制图源码 make_figures.py、早期报告生成脚本 make_docx.py、特征分析辅助脚本、原始提取与对齐代码、配置、锁定依赖及模型来源清单。question1_submission.zip 保留为竞赛技术附件的第一问原始封装。另附 12 页代码架构与模型详解 PDF。

原始MP4视频和大型预训练权重未进入 Git。前者属于赛题原始数据；后者可由 prepare_models.py 根据 models/model_manifest.json 中的模型仓库、修订号与文件校验值准备。缺少原始视频或权重时，可以阅读和核验已保存的结果文件，但不能从头重新提取全部特征。paper_work/build_paper.py 保留原始 Windows 生成环境路径；在 Linux 上重新生成 Word 需调整其路径与字体设置。已有 Word 和 PDF 可直接阅读。

问题一实测覆盖100/100条视频，共1931个词级位置；语音有效1931词、视觉有效1601词。上述是处理覆盖与结构核验，不能写成人工逐词对齐准确率。详细方法、参数与局限见[完整Word报告](问题一_分析与数学建模_完整版.docx)及[第一问结果报告](results/第一问结果报告.md)。
