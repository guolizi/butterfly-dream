# 🦋 Butterfly Dream — 多媒体记忆支持

## 概述

为 Butterfly Dream 增加图片、音频、视频等多媒体内容的存储和检索能力，实现「文本事实 + 媒体附件」的联合记忆。

## 背景

当前 Butterfly Dream 只支持纯文本事实的存储和三维检索（FTS5 + Jaccard + HRR 向量 + 场景权重）。用户在使用过程中会分享图片、语音消息、视频等媒体内容，系统需要记住这些内容并在后续对话中能通过关键词或语义检索到。

## 架构

```
fact ← 1:N → media_attachments
  │              ├── file_path (相对路径, CAS存储)
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
    created_at    TEXT DEFAULT (datetime('now'))
);

-- FTS5 全文索引（让 media 描述/字幕/转写可搜索）
CREATE VIRTUAL TABLE media_attachments_fts USING fts5(
    description, caption, transcript,
    content=media_attachments, content_rowid=media_id
);
```

### 索引

```sql
CREATE INDEX idx_media_fact    ON media_attachments(fact_id);
CREATE INDEX idx_media_sha256  ON media_attachments(sha256) WHERE sha256 != '';
CREATE INDEX idx_media_path    ON media_attachments(file_path);
CREATE INDEX idx_media_mime    ON media_attachments(mime_type);
CREATE INDEX idx_media_created ON media_attachments(created_at DESC);
```

### 与现有表的交互

- **merge_log**：语义合并时，新内容被吸收到已有 fact，不产生新的 fact_id，无需 re-parent
- **entity_relations**：媒体本身不创建实体关系，但父 fact 的实体关系保持不变
- **CASCADE**：`PRAGMA foreign_keys=ON` 启用后，删除 fact 自动清理其 media_attachments 行

## 文件存储

内容寻址存储（Content-Addressable Storage, CAS），使用 SHA-256 哈希作为文件名和子目录分片：

```
{media_dir}/
├── im/              ← image/*
│   └── {sha[:2]}/
│       └── {sha256}.{ext}
├── au/              ← audio/*
├── vi/              ← video/*
|    ├── ot/              ← other (application/octet-stream 等)
|    └── thumbs/          ← 缩略图 (自动生成, JPEG 格式)
|        ├── im/
|        └── ...
```

**路径规则**：
- 类型编码：`image/` → `im`，`audio/` → `au`，`video/` → `vi`，其他 → `ot`
- 文件名：完整 `sha256.{ext}`（自动去重）
- 扩展名：从 MIME 类型解析并映射（`jpeg→jpg`, `mpeg→mp3`, `svg+xml→svg` 等）
- 路径安全：使用 `realpath` 验证路径在 `media_dir` 内，防止 `../../../etc/passwd`

### 压缩存储

压缩默认开启，可在 YAML 配置中关闭或调整参数：

```yaml
plugins:
  butterfly-dream:
    compression:
      enabled: true          # 总开关，默认开启
      max_size_mb: 100       # 超过此大小(MB)的文件跳过压缩，防止卡死
      timeout: 600           # ffmpeg 超时秒数（默认10分钟）
      image:
        quality: 85           # JPEG quality (1-100)
        max_dim: 1920         # 超过此尺寸则缩放
        convert_to_jpeg: true # PNG/GIF → JPEG 有损压缩
      video:
        bitrate: "1M"         # ffmpeg -b:v
        max_fps: 30
        max_dim: 1280
        audio_bitrate: "128k"
      audio:
        bitrate: "128k"       # ffmpeg -b:a
        sample_rate: 44100
```

**压缩策略**：
- **图片**：用 Pillow 转为 JPEG（quality 可调），检测到 PNG 等格式自动转换并重写 MIME 为 `image/jpeg`
- **视频**：用 ffmpeg 转码为 H.264/AAC MP4，码率和分辨率可调
- **音频**：用 ffmpeg 转码为 MP3，比特率和采样率可调
- **智能跳过**：如果压缩后体积没减少（如已压缩的 JPEG/小文件），自动保留原始文件
- **大文件保护**：超过 `max_size_mb`（默认 100MB）的文件跳过压缩，避免长时间卡顿
- **可配超时**：`timeout` 控制 ffmpeg 最大等待时间（默认 10 分钟），超时自动回退到原始文件
- **哈希变更**：压缩后的 SHA-256 基于压缩版本，去重也是在压缩后内容上

**关闭压缩**：`compression: { enabled: false }` 即可恢复原始存储行为。不传 `compression` 参数也不会压缩（向后兼容）。

**缩略图**（`media_utils.py`）：
- 仅对 `image/*` 且 >50KB 的文件自动生成
- 最大 320×240，JPEG 格式，quality=75
- 存在 `thumbs/{原路径}.jpg` 下，同名文件复用

## 检索管道改造

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
返回带附件的 facts（含 media: [...] 字段）
```

**关键变更点**：
1. `_fts_candidates` 增加 `media_attachments_fts` 的并行搜索
2. 媒体匹配到的 description/transcript 提升父 fact 的 relevance 评分
3. 返回结果中增加 `media` 字段和 `_media_match` 标志
4. 媒体结果对应的父 fact 即使没被 `facts_fts` 匹配也会包含在结果中

### HRR 向量集成

媒体 `description` 在写入时编码到父 fact 的 `hrr_vector` 中，使得代数检索（probe/reason）也能命中包含媒体描述的事实。

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

列出磁盘上有但 DB 中无引用的孤儿文件。

### 工具：`media_cleanup`

删除孤儿文件。支持 `dry_run=True`（预览）和 `dry_run=False`（实际删除）。

## 实现路线

| 阶段 | 内容 | 涉及文件 | 状态 |
|:----|:----|:---------|:----:|
| **P0** | Schema + FTS5 + CAS文件存储 + attach/detach + 路径安全 | `store.py` | ✅ |
| **P1** | 检索管道改造（并行FTS5 + 三维评分）+ HRR 向量集成 | `retrieval.py`, `store.py` | ✅ |
| **P2** | 工具操作 + 路径验证 + EXIF 剥离 | `__init__.py`, `media_utils.py` | ✅ (路径/工具, EXIF 待做) |
| **P3** | 缩略图 + 文件GC + 会话权限 + EXIF | `media_utils.py` | ✅ (缩略图+GC, 会话权限+EXIF 待做) |
| **P4** | 媒体压缩（图片/视频/音频，默认开启） | `media_compressor.py`, `store.py`, `__init__.py` | ✅ |

## 已知缺陷

### P0 — 已修复
1. ✅ **媒体描述不可检索**：`description/caption/transcript` 不进 FTS5 → 已修复，FTS5 同步触发器已添加
2. ✅ **文件孤儿**：`ON DELETE CASCADE` 只删数据库行 → 已修复，`PRAGMA foreign_keys=ON` + `media_orphans()` + `media_cleanup()` 三管齐下
3. ✅ **不用 BLOB**：已采用文件系统存储 + 数据库存路径 + SHA-256 的架构

### P2 — 待修复
4. ❌ **EXIF 数据泄露**：JPEG/HEIC 存储时未剥离 GPS 坐标 → 待实现
5. ❌ **URL SSRF**：`storage_type='url'` 只允许 `https://` → 待实现

### P3 — 待实现
6. ❌ **会话权限**：不同 session 的媒体隔离 → 待实现

### 不支持的特性
7. ❌ **媒体自动提取**：`on_pre_compress` / `sync_turn` 等生命周期钩子只提取文本事实，不会自动扫描消息中的图片/音频/视频附件并调用 `attach_media`。媒体需要 LLM 在对话中显式调用 `media_attach` 工具手动存储。

## 相关文件

- `src/butterfly_dream/store.py` — SQLite 存储层（media_attachments 表 + attach/detach/orphans/cleanup）
- `src/butterfly_dream/retrieval.py` — 检索管道（并行 FTS5 搜索）
- `src/butterfly_dream/media_utils.py` — 缩略图生成 + 文件 GC
- `src/butterfly_dream/media_compressor.py` — 媒体压缩（图片/视频/音频，Pillow + ffmpeg）
- `src/butterfly_dream/holographic.py` — HRR 编码引擎（不变）
- `src/butterfly_dream/__init__.py` — 插件入口（工具 handler）
- `tests/test_media_real.py` — 真实媒体文件端到端测试
- `tests/test_media_compression.py` — 14 个压缩功能测试
