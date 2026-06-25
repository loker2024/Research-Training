# Overleaf 使用说明

本目录是根据 `docs/report/experiment_paper.md` 和 `参考资料/模版和其他资料/小论文模板-2026.doc` 整理出的 Overleaf 项目。

## 文件说明

- `main.tex`：Overleaf 主文件，包含中英文题名、作者、单位、摘要、关键词和 LaTeX 导言区。
- `body_content.tex`：论文正文，由 Markdown 论文转换而来，包含章节、表格、图和参考文献。
- `figures/`：论文引用的实验图。
- `references.bib`：参考文献 BibTeX 条目，当前正文中仍保留手写参考文献列表；如后续改为 BibTeX，可直接复用。

## 上传到 Overleaf

1. 将本目录中的文件整体上传到 Overleaf，或直接上传 `../overleaf_project.zip`。
2. 在 Overleaf 左上角 `Menu` 中，将 Compiler 设置为 `XeLaTeX`。
3. 确认主文件为 `main.tex`。
4. 作者和学院信息已按“涂家俊；合肥工业大学计算机与信息学院，合肥 230601”写入，提交前可再核对英文姓名与英文学院名称是否符合课程要求。

## 注意事项

- 当前本地环境未安装 `xelatex`，因此已完成静态检查和资源检查，但未在本机编译 PDF。
- 预测曲线与残差图章节保留了分析口径，但没有引用旧 v1 图片；后续如果补齐 v2 checkpoint，可再生成正式预测/残差图插入正文。
