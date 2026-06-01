# 🦋 Butterfly Dream — 多媒体记忆支持

## 概述

为 Butterfly Dream 增加图片、音频、视频等多媒体内容的存储和检索能力，实现「文本事实 + 媒体附件」的联合记忆。

## 背景

当前 Butterfly Dream 只支持纯文本事实的存储和三维检索（FTS5 + Jaccard + HRR 向量 + 场景权重）。用户在使用过程中会分享图片、语音消息、视频等媒体内容，系统需要记住这些内容并在后续对话中能通过关键词或语义检索到。

## 架构

```
fact ← 1:N → media_attachments
  │              ├── file_path (相对路径)
  │              ├── mime_type
  │              ├── description → FTS5 可搜索
  │              ├── caption
  │              └── transcript (语音转文字)
  │
  └── 三维评分 (relevance × recency × importance) 继承到 media
```

文本事实携带多媒体附件。检索时 FTS5 同时搜 `facts_fts` 和 `media_attachments_fts`，三维评分以父 fact 为准。

## Schema 设计

新建 `media_attachments` 表，与 `facts` 表 1:N 关联：

```sql
CREATE TABLE media_attachments (
    media_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    fact_id       INTEGER NOT NULL REFERENCES facts(fact_id) ON DELETE CASCADE,
    storage_type  TEXT NOT NULL DEFAULT 'file'
                  CHECK(storage_type IN ('file', 'url')),
    file_path     TEXT NOT NULL,
    mime_type     TEXT NOT NULL,
    file_size     INTEGER NOT NULL DEFAULT 0,
    sha256        TEXT NOT NULL DEFAULT '',
    description   TEXT NOT NULL DEFAULT '',
    caption       TEXT DEFAULT '',
    transcript    TEXT DEFAULT '',
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- FTS5 全文索引（让 media 描述/字幕/转写可搜索）
CREATE VIRTUAL TABLE media_attachments_fts USING fts5(
    description, caption, transcript,
    content=media_attachments, content_rowid=media_id
);

CREATE INDEX idx_media_fact    ON media_attachments(fact_id);
CREATE INDEX idx_media_sha256  ON media_attachments(sha256) WHERE sha256 != '';
CREATE INDEX idx_media_mime    ON media_attachments(mime_type);
CREATE INDEX idx_media_created ON media_attachments(created_at DESC);
```

### 与现有表的交互

- **merge_log**：语义合并时，被吸收 fact 的媒体附件自动 re-parent 到保留 fact（`UPDATE media_attachments SET fact_id=?`）
- **entity_relations**：媒体本身不创建实体关系，但父 fact 的实体关系保持不变

## 文件存储

```
$HERMES_HOME/
  ├── butterfly_memory.db        ← SQLite
  └── media/
      ├── images/                ← image/jpeg, image/png, image/webp
      ├── audio/                 ← audio/ogg, audio/mp4, audio/wav
      └── video/                 ← video/mp4, video/webm
```

**路径规则**：
- 入库存相对 `$HERMES_HOME` 的路径（如 `media/images/abc123.jpg`）
- 运行时解析为绝对路径
- 文件名用 `sha256[:16]_timestamp.ext` 防冲突

## 检索管道改造

### 当前流程

```
FTS5 MATCH facts_fts → 三维评分 → 返回 facts
```

### 改造后流程

```
并行 FTS5 搜索 facts_fts + media_attachments_fts
        │
        ▼
UNION 结果集（按 fact_id 去重）
        │
        ▼
三维评分（以父 fact 为准）
        │
        ▼
LEFT JOIN media_attachments → 返回带附件的 facts
```

**关键变更点**：
1. `_fts_candidates` 增加 `media_attachments_fts` 的并行搜索
2. 媒体匹配到的 description/transcript 内容合并到父 fact 的 relevance 评分中
3. 返回结果中增加 `media: [...]` 字段

### HRR 向量集成

媒体 `description` 应在写入时编码到父 fact 的 `hrr_vector` 中：

```python
# store.py add_media() 时
if hrr_vector is not None:
    media_hrr = hrr.encode_text(description, self._hrr_dim)
    new_hrr = hrr.bundle(existing_hrr, media_hrr)  # bundle 进父 vector
```

这样代数检索（probe/reason）也能命中包含媒体描述的事实。

## 新增操作

### 工具：`media_attach`

```python
{
    "name": "media_attach",
    "description": "Attach a media file to an existing fact",
    "parameters": {
        "fact_id": {"type": "integer"},
        "file_path": {"type": "string"},
        "mime_type": {"type": "string"},
        "description": {"type": "string"},
        "caption": {"type": "string", "optional": True},
        "transcript": {"type": "string", "optional": True},
    }
}
```

### 工具：`media_detach`

从 fact 解除媒体附件关联（不删磁盘文件）。

### 工具：`media_orphans`

列出磁盘上有但 DB 中无引用的孤儿文件，支持清理。

## 实现路线

| 阶段 | 内容 | 涉及文件 | 工作量 |
|:----|:----|:---------|:------|
| **P0** | Schema + FTS5 + 文件存储 + 基本 attach/detach + 路径安全 | `store.py`, `__init__.py` | 1-2 天 |
| **P1** | 检索管道改造（并行 FTS5 + 三维评分） + HRR 集成 | `retrieval.py`, `store.py` | 1 天 |
| **P2** | 新工具操作 + 文件 GC + 路径验证 | `__init__.py` | 1 天 |
| **P3** | SHA-256 去重 + EXIF 剥离 + 缩略图 + 会话权限 | `media_utils.py` | 2 天 |

## 已知缺陷（P0 实施前需处理）

### P0 — 必须先修

1. **媒体描述不可检索**：`description/caption/transcript` 不进 FTS5 → 用户搜关键词匹配不到媒体
2. **文件孤儿**：`ON DELETE CASCADE` 只删数据库行，不删磁盘文件 → 需 GC 机制
3. **不用 BLOB**：SQLite BLOB 存媒体导致 WAL 爆炸、死锁、备份灾难

### P1 — 性能

4. **N+1 查询**：检索 10 个 facts 若逐个 JOIN 查 media → 1+10=11 次查询 → 改为 LEFT JOIN
5. **HRR 向量集成**：媒体 `description` 应 bundle 到父 fact 的 `hrr_vector`
6. **三维评分一致性**：媒体匹配的结果应继承父 fact 的 recency/importance

### P2 — 安全

7. **路径遍历攻击**：`file_path` 必须 `os.path.realpath` 验证在 `media/` 目录内
8. **URL SSRF**：`storage_type='url'` 只允许 `https://`
9. **EXIF 数据泄露**：JPEG/HEIC 存储时剥离 GPS 坐标
10. **路径可移植性**：存相对 `$HERMES_HOME` 路径

### P3 — 增强

11. **SHA-256 去重**：相同文件多次引用不重复存储
12. **缩略图**：大图预览前缩略
13. **会话权限**：不同 session 不互相看到对方媒体

## 相关文件

- `src/butterfly_dream/store.py` — SQLite 存储层（加 media 表）
- `src/butterfly_dream/retrieval.py` — 检索管道（加并行 FTS5）
- `src/butterfly_dream/holographic.py` — HRR 编码引擎（不变）
- `src/butterfly_dream/__init__.py` — 插件入口（加工具 handler）
