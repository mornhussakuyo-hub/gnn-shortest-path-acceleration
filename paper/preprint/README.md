# arXiv 预印本包

核对日期：2026-08-13。

## 文件

- `main.pdf`：公开预览稿。科学正文与 `paper/main_en.tex` 的 2026-08-13 修订版一致。
- `arxiv_source_2026-08-13.tar.gz`：arXiv 上传源文件包，顶层入口为 `main.tex`。
- `main.tex`、`main.bbl`、`references.bib`、`elsarticle.cls`、`elsarticle-harv.bst` 和
  `figures/`：上传包的可检查展开内容。

源文件包已在独立临时目录中仅依赖包内文件完成完整 pdfLaTeX/BibTeX 编译，生成 34 页 PDF；引用、
交叉引用和图片均完整。

## 发布前人工门

公开上传前仍需由三位作者共同确认：

1. 全体作者同意公开预印本并同意当前版本；
2. 确认选择公开优先权，即接受题名和作者公开后会削弱 EAAI 双匿名效果；
3. 检查数据、地图和图表的公开许可边界；
4. 确认利益冲突声明和生成式 AI 使用声明准确。

当前预印本保留三位作者和已确认的武汉大学人工智能学院单位；通讯作者、邮箱、ORCID 和致谢仍须
在正式期刊投稿前补齐。

## arXiv 提交流程

1. 登录或注册 arXiv；新账户或首次进入新分类可能需要 endorsement。
2. 选择 `Start New Submission`。
3. 建议主分类选 `cs.LG`，交叉分类选 `cs.DS`；最终分类由作者按论文定位确认。
4. 上传 `arxiv_source_2026-08-13.tar.gz`，确认顶层 TeX 文件为 `main.tex`。
5. 从 PDF/源文件填写标题、三位作者、摘要和关键词；Comments 可写
   `35 pages, 8 figures, preprint`。
6. 若没有基金或单位的强制许可要求，优先选择
   `arXiv.org perpetual, non-exclusive license 1.0`，不要选择 CC0。
7. 打开 arXiv 生成的 PDF，逐页核对首页作者、表格、公式、八幅图、参考文献和最后声明。
8. 完成提交并等待 arXiv moderation；获得编号后再更新仓库和后续 EAAI 投稿材料。

## 与 EAAI 的关系

Elsevier 当前共享政策允许作者在任何时间、任何地点共享 preprint，并建议文章接收后把预印本
链接更新到正式 DOI。该政策同时提醒双匿名期刊可能有单独规则，应以目标期刊作者指南为准。
EAAI 当前采用双匿名审稿；公开具名预印本会让检索者推断作者身份，因此在 EAAI 匿名正文中不要
加入能识别作者的预印本链接。投稿系统如询问既有预印本，应如实填写；投稿信可向编辑说明预印本
存在，但匿名正文仍保持无作者信息。

官方入口：

- Elsevier sharing policy: https://www.elsevier.com/about/policies-and-standards/sharing
- EAAI Guide for Authors: https://www.sciencedirect.com/journal/engineering-applications-of-artificial-intelligence/publish/guide-for-authors
- arXiv submission guide: https://info.arxiv.org/help/submit/index.html
- arXiv TeX guide: https://info.arxiv.org/help/submit_tex.html
- arXiv license guide: https://info.arxiv.org/help/license/index.html
