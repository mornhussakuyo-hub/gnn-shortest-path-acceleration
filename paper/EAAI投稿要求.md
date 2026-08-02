# EAAI 投稿要求与本项目准备清单

> 目标期刊：*Engineering Applications of Artificial Intelligence*（EAAI）
> 出版方：Elsevier；ISSN 0952-1976
> 信息核对日期：2026-08-02
> 用途：约束本项目英文稿、投稿附件和 Editorial Manager 提交流程。官网规则可能更新，正式提交当天应再次打开作者指南复核动态字段。

## 1. 先给结论

本项目与 EAAI 的主题范围总体匹配：期刊明确接收 AI 在真实工程问题中的创新应用，并将
“智能交通系统与智能车辆”“复杂网络”“神经网络与深度学习的真实应用”“决策支持系统”列为
关注方向。本文同时具备交通工程任务、AI 排序方法、真实道路图、两城验证和在线部署评测，具备
投稿基础。

当前最大的投稿风险不是版式，而是以下四点：

1. 摘要必须把“AI 方面的新贡献”和“工程应用”分别说清楚，不能只讲系统结果；
2. Porto、Chicago 数据及处理结果需要给出可访问来源、许可证或无法公开的理由，以满足期刊对
   公共数据和可复现性的强调；
3. 英文稿需要严格收敛主张，突出 BRIDGE/BRIDGE-B 的稳定排序增量，同时如实保留 Chicago
   在线工作量没有超过 Z0 的结果；
4. 作者单位、通讯作者、邮箱、基金、利益冲突、CRediT 作者贡献、数据/代码可用性和生成式 AI
   使用声明仍需补齐。

## 2. 期刊范围与 desk reject 条件

### 2.1 收稿范围

EAAI 要求论文报告 AI 的新颖方面，并将其用于真实工程应用；期刊同时强调用公共数据集验证，以便
复现实验。适合的稿件不应只是把现成模型套到一个数据集上，也不应只有 AI 算法而缺少可信的工程
问题、工程评价和实际意义。

本项目应在标题、摘要、引言和投稿信中保持同一定位：

- 工程问题：固定存储/预处理预算下，如何为精确道路最短路查询选择值得物化的区域；
- AI 贡献：历史 OD 驱动的双向需求传播、冻结 Z0 上的神经排序残差，以及预算感知的集合部署目标；
- 工程验证：Porto 与 Chicago、当前与未来窗口、严格非重叠部署、C++ 同实现延迟和 100% 距离正确率；
- 适用边界：神经残差稳定改善全局排序，但不是在所有城市、所有 Top-K 或在线指标上全面替代 Z0。

### 2.2 官网明列的四项 desk reject 条件

EAAI 官网明确说明，不满足以下条件的稿件会在外审前被退稿：

1. 新的“隐喻式”元启发式算法通常不予接收；具体边界以作者指南的 Article types 小节为准；
2. 摘要必须清楚说明 AI 贡献是什么、工程应用是什么；
3. 标题和摘要禁止使用未定义的缩写；
4. 稿件必须采用单栏格式。

本项目不属于隐喻式元启发式，但仍需执行以下检查：

- 标题尽量不使用 OD、CRP、GNN 等缩写；
- 摘要首次出现 OD、CRP、Z0、BRIDGE、BRIDGE-B 时必须定义；若缩写不能在一句话内清楚定义，
  直接改用全称；
- 英文初投稿使用单栏，不能提交当前双栏预览版；
- 摘要用明确句式分别写出 `AI contribution` 和 `engineering application`，不要让编辑自行推断。

## 3. 稿件类型、语言与整体格式

### 3.1 稿件类型

本项目应选择常规完整研究论文，对应投稿系统中的 **Full Length Article / Research Paper** 一类；
最终名称以 Editorial Manager 当天的下拉菜单为准。不要误选专题专刊（VSI）条目，除非确实响应
某个仍开放的征稿且正文针对该专题。

### 3.2 语言与文件

- 稿件使用规范英文；提交前进行拼写和语法检查；
- 初投可以使用 Word 或 LaTeX。LaTeX 建议使用 Elsevier 的 `elsarticle.cls` 与 BibTeX；
- EAAI 的单栏要求优先于任何双栏预览习惯；
- 正文按 `1`、`1.1`、`1.1.1` 编号，摘要不编号；
- 初投阶段参考文献重点是完整、准确和全文一致；接受后再严格服从期刊排版样式；
- 官网当前要求所有投稿不超过 50 页。文章类型页面同时写有文件小于 100 MB，而提交检查表又要求
  主文章文件严格小于 50 MB；为避免技术退稿，本项目统一执行更严格的“正文不超过 50 页、主稿
  小于 50 MB”，并在提交当天再次核对系统提示。

本仓库最终应另建 EAAI 英文入口，不直接覆盖当前中文 `main.tex`。建议保留中文稿作为证据底稿，
英文稿改用 `elsarticle` 单栏模板。

## 4. 标题页、摘要与关键词

### 4.1 标题页

标题页至少准备：

- 简洁、可检索且不含未定义缩写的英文标题；
- 每位作者的规范英文姓名；
- 作者单位及完整英文地址，用上标字母关联作者与单位；
- 明确的通讯作者、长期有效邮箱和必要联系信息；
- 作者当前地址与研究单位不同时，可另列 present/permanent address；
- ORCID 建议全部确认并绑定投稿账户。

EAAI 采用 **double anonymized review（双匿名审稿）**：作者与审稿人的身份相互隐藏。投稿时必须
分开提交含作者信息的标题页和不含作者姓名、单位、致谢等识别信息的匿名正文。标题页至少包含
题名、作者、单位、通讯作者完整地址与邮箱、致谢和利益冲突声明；匿名正文保留参考文献、图和表。

### 4.2 摘要

按不超过约 250 个英文词控制，写成一个可独立阅读的段落，不放参考文献，避免不常用缩写。正式
提交前需再以作者指南/系统显示的即时上限为准。

建议用五句逻辑完成：

1. 真实工程问题与资源约束；
2. 现有方法的缺口；
3. AI 方法贡献（Z0、BRIDGE、BRIDGE-B 中真正属于本文的部分）；
4. 两城、跨时间、在线精确评测的主要结果，同时保留不利结果；
5. 工程意义和部署边界。

必须显式出现类似下面的区分，但不必机械使用小标题：

- `The AI contribution is ...`
- `The engineering application is ...`

### 4.3 关键词

- 提供 **1–6 个英文关键词**；
- 避免用 `and`、`of` 连接成很长的复合词；
- 尽量使用领域通行、便于检索的单词或短语；
- 本项目可从以下候选中选 6 个以内：`shortest-path index`、`road network`、
  `workload-aware indexing`、`graph neural network`、`learning to rank`、
  `intelligent transportation system`。

## 5. 正文结构与技术表达

完整研究论文建议组织为：

1. Introduction；
2. Related work；
3. Problem formulation；
4. Proposed methods；
5. Experimental protocol；
6. Results；
7. Discussion；
8. Limitations；
9. Conclusion；
10. 投稿声明与 References。

具体要求：

- 引言明确工程问题、AI 缺口、本文贡献和适用边界；
- 方法必须可复现：图构造、H/Y/F 时间切分、候选生成、标签、空间隔离、损失、优化器、种子、
  checkpoint 规则和部署选择过程都要给出；
- 结果不能只报最优种子；报告重复种子、波动、负结果和预先冻结的选择规则；
- 公共数据、代码、配置和结果摘要应使用持久链接或 DOI；
- 数学变量使用斜体，函数/算子按统一约定排版；公式作为可编辑文本而非图片；
- 使用 SI 单位；自定义单位或工作量指标首次出现时定义；
- 表格必须为可编辑文本，避免与正文重复陈述同一组数字；
- 每幅图、每张表都要在正文中按顺序引用并具有自解释标题/图注；
- 他人图表、地图底图、图标和数据若受版权限制，投稿前取得许可并正确署名。

## 6. Highlights、图文摘要和插图

### 6.1 Highlights

按 EAAI 投稿包的必备材料准备独立 Highlights 文件：

- 3–5 条；
- 每条不超过 85 个字符，字符数包含空格；
- 强调新方法和主要结果；
- 面向一般读者，尽量不使用行话、缩写和首字母简称；
- 独立 Word 文件上传，并在文件类型中选择 `Highlights`。

官方通用说明称 Highlights 最迟在 final files 阶段必须提供；考虑 EAAI 的投稿检查和减少技术退回，
本项目应在首次投稿时一并上传。

### 6.2 Graphical abstract

图文摘要通常属于鼓励项；若 Editorial Manager 将其标为必需，则按系统要求上传独立文件。本项目
已有方法流程图，可重新排成适合图文摘要的横向单图，重点表现：

`historical OD → bidirectional demand propagation → residual ranking → non-overlapping deployment → exact queries`。

不要使用生成式 AI 或 AI 辅助工具制作或修改论文图、数据图和图文摘要；EAAI 当前作者指南对此
采用比 Elsevier 通用政策更严格的口径。方法流程图使用人工矢量绘制，数据图必须直接来自底层数据
和可复现脚本，严禁生成或改写实验数据。

### 6.3 插图交付

- 优先提交矢量 PDF/EPS；照片或连续色调图使用 TIFF/JPEG，线图和混合图保证足够分辨率；
- 字体、线宽、字号和配色保持一致，缩放到最终尺寸后仍能阅读；
- 彩色图同时检查灰度可辨性和色觉无障碍；
- 每幅图提供独立图注；图内文字尽量短，解释放在图注中；
- LaTeX 源文件、BibTeX、全部图片和必要样式文件必须能在干净环境完整编译。

## 7. 参考文献与研究对象引用

- 文中每条引用必须出现在参考文献表，参考文献表每条也必须在正文引用；
- 核对作者、年份、题名、期刊/会议、卷期页码和 DOI；
- 数据集、软件和代码仓库应作为正式研究对象引用，不应只在正文放裸链接；
- 预印本若已有最终同行评审版本，应优先引用正式版本；
- 初投参考文献格式保持一致即可，后续按 EAAI 样式统一；`elsarticle` 与 BibTeX 是官方推荐路线；
- 不得使用无法核验的生成式 AI 引用；所有文献需逐条回到原文或权威索引确认。

## 8. 署名、伦理与必备声明

### 8.1 作者与贡献

- 所有作者必须对研究有实质贡献、批准最终稿并同意投稿；
- 投稿前冻结作者顺序、通讯作者和贡献，审稿期间变更作者通常需要全体书面同意和编辑批准；
- 使用 CRediT 角色写作者贡献，例如 Conceptualization、Methodology、Software、Validation、
  Formal analysis、Data curation、Visualization、Writing、Supervision；
- 生成式 AI 工具不得列为作者。

### 8.2 利益冲突与基金

- **无论是否存在冲突，都必须提供 competing interests statement**；
- 逐项报告可能影响研究判断的财务或个人关系；无冲突时使用期刊认可的无冲突表述；
- 列出基金机构、项目编号及基金方在研究设计、数据、写作和投稿决定中的角色；无外部资助也应
  明确说明。

### 8.3 数据与代码

投稿时准备 Data availability statement，并明确：

- Porto、Chicago 原始数据来源、公开 URL/DOI、许可证与访问日期；
- 处理后数据、候选、split、标签清单、评测查询和摘要结果是否公开；
- 训练/评测代码、环境、配置和随机种子如何获取；
- 第三方条款不允许再分发的数据，说明限制、申请方式和可公开的替代材料；
- 若暂时无法公开，给出真实原因，不能把“可向作者索取”当作默认替代方案。

EAAI 特别强调公共数据验证和易复现性，因此本项目的公开材料不是普通加分项，而是编辑初筛时的
核心证据。公开仓库还应排除服务器凭据、日志密码和无授权数据。

### 8.4 生成式 AI 声明

Elsevier 的作者政策在 2026 年 6 月更新。若生成式 AI 对稿件结构、句子组织、翻译或内容表达进行
了实质性辅助，应在参考文献前放独立声明，写明工具/服务名称、用途、人工复核，并由作者承担全部
责任。只做基础拼写、语法和标点检查通常不必声明。

推荐按官方模板改写：

> During the preparation of this work, the authors used [TOOL/SERVICE] to [PURPOSE]. After using
> this tool/service, the authors reviewed and edited the content as needed and take full
> responsibility for the content of the publication.

AI 若属于研究方法的一部分，应在 Methods 中可复现地报告工具/模型名称、版本、开发者和用途，
不能只放在写作声明中。

### 8.5 原创性与重复投稿

- 稿件不得同时投往其他期刊；
- 保证原创，正确引用他人文字、方法、图表和数据；
- 预印本通常不被 Elsevier 视为重复发表，但提交前仍应核对 EAAI 当日政策并在系统中如实申报；
- 不得拆分同一研究为实质重复论文，也不得隐瞒相关在投/已发表工作。

## 9. 投稿文件包

首次投稿前至少准备以下文件或信息：

| 项目 | 状态要求 | 本项目当前情况 |
| --- | --- | --- |
| 单栏英文匿名主稿 | 必备 | 尚未形成 EAAI 英文版；当前中文单栏匿名稿可作翻译底稿 |
| 独立标题页与作者信息 | 必备 | 中文标题页已建；英文名、单位、地址、通讯作者、邮箱、ORCID 待确认 |
| Abstract 与 1–6 keywords | 必备 | 中文摘要已有；需按 EAAI 规则重写英文摘要并收敛关键词 |
| Highlights 独立文件 | 按必备准备 | 已建立英文文本，翻译全文后再核对措辞 |
| Graphical abstract | 建议准备，系统标必需时上传 | 可由现有方法流程图改制 |
| Cover letter | 强烈建议 | 尚未制作 |
| Competing interests statement | 必备，即使无冲突 | 正文只有空白占位 |
| Funding statement | 必备信息 | 正文只有空白占位 |
| CRediT author statement | 按必备准备 | 正文只有空白占位 |
| Data availability statement | 按必备准备 | 已有审计口径；永久链接待补 |
| Code availability / repository | 强烈建议 | 需确定公开范围、归档版本和永久链接 |
| Generative AI declaration | 有实质使用时必备 | 中文稿已按实际用途加入，投稿前由全体作者确认英文表述 |
| 图表源文件与图注 | 必备 | 已有正式图，但需做英文版、版权和最终分辨率检查 |
| Supplementary material | 视需要 | 可放扩展消融、额外地图、实现细节和复现清单 |
| 推荐/回避审稿人 | 系统要求时提供 | 尚未准备；推荐人不得存在利益冲突 |

Cover letter 建议一页内完成：说明文章为何符合 EAAI、分别概括 AI 贡献与工程应用、指出公共数据
和代码计划、声明未一稿多投并经全体作者同意；不要重复粘贴摘要，也不要使用夸张的“首次”“全面
领先”等不可审计表述。

## 10. 审稿与发表方式

- 投稿入口为 [EAAI Editorial Manager](https://www.editorialmanager.com/eaai/default.aspx)；
- 期刊采用双匿名审稿；编辑先做范围与质量初筛，合适稿件通常交给至少两位独立专家；
- 编辑负责最终决定；修改稿应提交逐条 response to reviewers，并标明正文改动；
- 期刊为混合发表模式：选择订阅发表时官网显示作者不缴 publication fee；
- 截至 2026-08-02，官网显示 Gold Open Access APC 为 **USD 3,040（未含税）**，实际金额可能因
  协议、减免和地区而变化；应在接受后选择许可前重新核价，不能把当前数字写进长期预算而不复核。

## 11. 本项目投稿前的执行顺序

1. 确认三位作者英文姓名、单位、作者顺序、通讯作者、邮箱、ORCID、基金和 CRediT 分工；
2. 冻结论文结果口径，确保 README、阶段报告、论文表格和 PDF 数字一致；
3. 核实 Porto/Chicago 数据的公共来源、许可证与可再分发边界，设计代码/数据归档；
4. 建立 `elsarticle` 单栏英文稿，先重写标题、摘要、关键词和贡献，再翻译正文；
5. 把 AI 贡献与工程应用的对应关系贯穿摘要、引言、方法、讨论和投稿信；
6. 完成英文图表、Highlights、图文摘要、Cover letter 和所有声明；
7. 用干净环境编译投稿源文件，检查引用、图表、链接、缩写、单位和补充材料；
8. 全体作者逐页审阅并确认最终稿、作者顺序、数据声明和生成式 AI 声明；
9. 提交当天重新打开 EAAI 作者指南，复核文章类型、摘要上限、附件必选状态、APC 和系统字段；
10. 在 Editorial Manager 生成合并 PDF 后再次逐页检查，再完成最终提交。

## 12. 官方来源

以下链接均应在正式提交当天重新查看：

1. [EAAI Guide for Authors（核心规则）](https://www.sciencedirect.com/journal/engineering-applications-of-artificial-intelligence/publish/guide-for-authors)
2. [EAAI 期刊范围与四项 desk reject 条件（Elsevier）](https://shop.elsevier.com/journals/engineering-applications-of-artificial-intelligence/0952-1976)
3. [EAAI ScienceDirect 主页、开放获取与费用](https://www.sciencedirect.com/journal/engineering-applications-of-artificial-intelligence)
4. [EAAI Editorial Manager 投稿入口](https://www.editorialmanager.com/eaai/default.aspx)
5. [Elsevier 作者伦理与政策总页](https://www.elsevier.com/researcher/author/policies-and-guidelines)
6. [Elsevier Highlights 说明](https://www.elsevier.com/researcher/author/tools-and-resources/highlights)
7. [Elsevier 生成式 AI 期刊政策（2026-06 更新）](https://www.elsevier.com/about/policies-and-standards/generative-ai-policies-for-journals)
8. [Elsevier 研究数据政策](https://www.elsevier.com/about/policies-and-standards/research-data)
9. [Elsevier Data statement 说明](https://www.elsevier.com/researcher/author/tools-and-resources/research-data/data-statement)
10. [Elsevier LaTeX 说明](https://www.elsevier.com/researcher/author/policies-and-guidelines/latex-instructions)

第三方期刊指标、经验审稿周期和模板网站未作为硬性要求来源；若其内容与上述官方页面冲突，以
EAAI 作者指南和 Editorial Manager 当日显示为准。
