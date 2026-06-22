# 剩余工作待办清单

更新时间：2026-06-14 12:56:17 +08:00

## 当前已完成

- [x] 第 8 项：检查并提交未提交的文档路径、报告引用和结果索引校准变更
- [x] ETTh1/ETTm1 正式核心实验：2 数据集 × 5 步长 × 5 模型，共 50 组
- [x] Autoformer/PatchTST 消融实验：2 数据集 × 2 步长 × 4 消融模型，共 16 组
- [x] 第 6 步可视化与深入分析：指标趋势图、最优模型图、复杂度图、消融图、预测曲线和残差图
- [x] 第 7 步报告/论文草稿：`docs/report/experiment_report_demo.md` 与 `docs/report/experiment_paper.md`
- [x] MAPE 指标汇总与报告补充：生成正式实验 MAPE 明细/聚合表，并补充报告中的 MAPE 说明
- [x] 单变量 vs 多变量对比实验：新增专用训练脚本、流程 notebook，并完成 h96/h336 代表性快速对比；正式全量实验尚未进行

## 必须优先补齐

- [x] 数据集范围确认
  - [x] 当前项目正式实验和报告只采用 ETTh1、ETTm1 两个数据集
  - [x] ECL 不再作为必须补齐的正式或附录实验

- [x] 单变量 vs 多变量对比实验
  - [x] 明确单变量口径：只输入目标列并预测目标列
  - [x] 优先选择 ETTh1/ETTm1 的 h96 和 h336 做代表性对比
  - [x] 至少覆盖 `lstm,transformer,autoformer,patchtst`，资源允许再补 Informer
  - [x] 生成单变量/多变量对比表，并补充到报告实验设置与讨论部分
  - [ ] 正式全量实验尚未进行：需将 `sample_limit` 改为 `0`，建议 `epochs=20`、`patience=5` 后重跑

- [x] MAPE 指标汇总与报告补充
  - [x] 从已有 summary JSON 中提取 `MAPE` 与 `MAPE_target`
  - [x] 在 `results/v1_csv/` 和 `results/v1_md/` 中生成含 MAPE 的补充表
  - [x] 在报告中补充 MAPE 说明；若不作为主指标，说明 MAPE 对接近零值序列敏感

## 可增强但非立即必须

- [ ] 多随机种子稳定性实验
  - [ ] 选定代表性组合：ETTh1/ETTm1 × h96/h336 × Autoformer/PatchTST
  - [ ] 使用至少 3 个 seed，报告均值和标准差
  - [ ] 更新论文“局限性”或“稳健性分析”

- [ ] 验证集 Top1 LSTM/Transformer 正式对比
  - [ ] 使用 `configs/lstm_top1.json` 跑 ETTh1/ETTm1 五步长正式实验
  - [ ] 使用 `configs/transformer_top1.json` 跑 ETTh1/ETTm1 五步长正式实验
  - [ ] 与 `formal_seed42` 中原 LSTM/Transformer、Autoformer、PatchTST 统一汇总对比

- [ ] 外生变量融合说明或扩展
  - [ ] 若不实现，报告中明确当前实验使用多变量历史序列输入，未单独引入外部协变量
  - [ ] 若实现，设计时间特征或外部变量输入，并加入一组小规模对比

- [ ] 概率预测说明或扩展
  - [ ] 若不实现，报告中明确当前范围为点预测
  - [ ] 若实现，优先使用分位数损失输出 P10/P50/P90，并增加区间覆盖率指标

## 最终交付整理

- [ ] 补全论文作者、单位、课程信息
- [ ] 检查报告图片路径、结果表路径和附录文件路径
- [ ] 根据课程模板决定是否导出 DOCX/PDF
- [ ] 检查 `git status`，确认没有遗留未提交变更
- [ ] 推送本地提交到远端仓库
