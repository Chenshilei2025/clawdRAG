# MM-RAG Agent 架构图 (Mermaid 版本)

## 1. 系统总体架构

```mermaid
graph TB
    subgraph "用户层 User Layer"
        CLI[CLI Client]
        Web[Web Browser]
        API[API Client]
    end

    subgraph "API 网关层 API Gateway"
        Auth[Authentication]
        Rate[Rate Limiting]
    end

    subgraph "消息总线层 Message Bus"
        Queue[asyncio.Queue]
        Inbound[InboundQueue]
        Outbound[OutboundQueue]
    end

    subgraph "主代理层 Main Agent"
        MainAgent[MainAgent<br/>PEO Loop]
        Context[ContextBuilder]
        Memory[MemoryConsolidator]
        Session[SessionManager]
    end

    subgraph "工具层 Tool Layer"
        SpawnTool[SubagentManagerTool<br/>spawn_subagent]
        VectorSearch[VectorSearchTool]
        Embedding[EmbeddingGeneratorTool]
        OCR[ImageOCRTool]
        Caption[ImageCaptioningTool]
        FileTool[FileSystem Tools]
    end

    subgraph "子代理层 Subagent Layer"
        QueryRewriter[QueryRewriterSubagent]
        DocAnalyzer[DocumentAnalyzerSubagent]
        VideoAnalyzer[VideoAnalyzerSubagent]
        ImageAnalyzer[ImageAnalyzerSubagent]
    end

    subgraph "提供者层 Provider Layer"
        LLM[LLMProvider<br/>OpenAI/Anthropic/Qwen]
        Vision[VisionProvider<br/>GPT-4V/Claude-Vision]
        OCRProv[OCRProvider<br/>Tesseract/PaddleOCR]
        Emb[EmbeddingProvider<br/>CLIP/OpenAI]
    end

    subgraph "存储层 Storage Layer"
        VectorDB[Vector Store<br/>ChromaDB/Milvus]
        FileStore[File Storage<br/>Uploads/Cache]
        MemStore[Memory Store<br/>MEMORY.md/HISTORY.md]
        SessionStore[Session Store]
    end

    CLI --> Web
    Web --> API
    API --> Auth
    Auth --> Rate
    Rate --> Queue

    Queue --> Inbound
    Inbound --> MainAgent

    MainAgent --> Context
    MainAgent --> Memory
    MainAgent --> Session

    MainAgent --> SpawnTool
    MainAgent --> VectorSearch
    MainAgent --> Embedding
    MainAgent --> OCR
    MainAgent --> Caption
    MainAgent --> FileTool

    SpawnTool --> QueryRewriter
    SpawnTool --> DocAnalyzer
    SpawnTool --> VideoAnalyzer
    SpawnTool --> ImageAnalyzer

    QueryRewriter --> LLM
    DocAnalyzer --> LLM
    VideoAnalyzer --> LLM
    ImageAnalyzer --> LLM
    VectorSearch --> VectorDB
    Embedding --> Emb
    OCR --> OCRProv
    Caption --> Vision

    Context --> MemStore
    Session --> SessionStore
    FileTool --> FileStore

    MainAgent --> Outbound
    Outbound --> Queue
```

---

## 2. PEO 循环流程图

```mermaid
stateDiagram-v2
    [*] --> Plan: 用户输入

    Plan --> Plan: 理解意图
    Plan --> Plan: 分析上下文
    Plan --> Plan: 制定计划

    Plan --> Execute: 需要执行操作

    Execute --> Execute: 调用工具
    Execute --> Execute: 调用子代理
    Execute --> Execute: 执行操作

    Execute --> Observe: 获得结果

    Observe --> Observe: 观察结果
    Observe --> Observe: 评估效果
    Observe --> Plan: 需要继续

    Observe --> Learn: 任务完成

    Learn --> Learn: 更新记忆
    Learn --> Learn: 优化策略
    Learn --> [*]: 返回响应
```

---

## 3. 多模态处理流水线

```mermaid
flowchart LR
    Input([输入文件]) --> Detect{文件类型检测}

    Detect -->|PDF| PDF[PDFParser<br/>文本提取<br/>图像提取<br/>结构解析]
    Detect -->|PPT| PPT[PPTParser<br/>幻灯片提取<br/>内容解析]
    Detect -->|Image| ImgProc[图像处理<br/>OCR识别<br/>视觉理解]
    Detect -->|Video| VidProc[视频处理<br/>帧提取<br/>音频转录]
    Detect -->|Audio| AudProc[音频处理<br/>语音识别]

    PDF --> Embed[向量化<br/>Text Embedding]
    PPT --> Embed
    ImgProc --> ImgEmb[向量化<br/>CLIP Image Embedding]
    VidProc --> VidEmb[向量化<br/>Frame Embedding]
    AudProc --> TextEmb[向量化<br/>Text Embedding]

    Embed --> VectorDB[(向量存储<br/>ChromaDB)]
    ImgEmb --> VectorDB
    VidEmb --> VectorDB
    TextEmb --> VectorDB

    VectorDB --> Search[语义检索]
```

---

## 4. MainAgent 与 Subagent 交互

```mermaid
sequenceDiagram
    participant User as 用户
    participant Main as MainAgent
    participant Spawn as SpawnTool
    participant Factory as SubagentFactory
    participant Sub as Subagent
    participant LLM as LLM Provider
    participant Tools as Tools

    User->>Main: 发送查询

    Main->>Main: PEO - Plan
    Main->>Main: 分析查询模糊

    Main->>Spawn: spawn_subagent("query_rewriter")

    Spawn->>Factory: create_subagent("query_rewriter")

    Factory->>Sub: 实例化 QueryRewriterSubagent
    Factory-->>Spawn: 返回 Subagent

    Spawn->>Sub: execute({query, context, history})

    Sub->>LLM: 生成优化查询
    LLM-->>Sub: 返回结果

    Sub-->>Spawn: 返回优化后的查询
    Spawn-->>Main: 返回结果

    Main->>Main: PEO - Execute (继续)
    Main->>Spawn: spawn_subagent("video_analyzer")

    Spawn->>Factory: create_subagent("video_analyzer")
    Factory-->>Spawn: VideoAnalyzerSubagent

    Spawn->>Sub: execute({file, parameters})

    Sub->>Tools: FrameExtractor
    Tools-->>Sub: 帧列表

    Sub->>Tools: AudioTranscriber
    Tools-->>Sub: 转录文本

    Sub->>LLM: 分析内容
    LLM-->>Sub: 分析结果

    Sub-->>Spawn: 返回分析结果
    Spawn-->>Main: 返回结果

    Main->>Main: PEO - Observe
    Main->>Main: PEO - Learn (更新记忆)

    Main-->>User: 返回最终响应
```

---

## 5. 跨模态检索流程

```mermaid
flowchart TB
    Query([用户查询]) --> Rewrite[QueryRewriterSubagent<br/>查询优化]

    Rewrite --> Analyze{查询分析}

    Analyze -->|纯文本| TextSearch[文本向量检索<br/>Text Embedding]
    Analyze -->|纯图像| ImageSearch[图像向量检索<br/>CLIP Image Embedding]
    Analyze -->|混合| HybridSearch[多路检索<br/>Text + Image + Video]

    TextSearch --> VectorDB[(Vector Database)]
    ImageSearch --> VectorDB
    HybridSearch --> VectorDB

    VectorDB --> Results[候选结果]

    Results --> Fusion[结果融合<br/>Score Fusion]

    Fusion --> Rerank[重排序<br/>LLM-based Re-ranking]

    Rerank --> Dedup[去重<br/>Deduplication]

    Dedup --> Final([最终结果])

    Final --> MainAgent[MainAgent<br/>生成回答]
```

---

## 6. 子代理类图

```mermaid
classDiagram
    class Subagent {
        <<abstract>>
        +subagent_id: str
        +config: dict
        +llm_provider: LLMProvider
        +initialize()*
        +execute(parameters)*
        +cleanup()*
    }

    class SimpleSubagent {
        <<abstract>>
        +task_description: str
        +execute(parameters)*
    }

    class QueryRewriterSubagent {
        +task_description: str
        +execute(parameters)
    }

    class DocumentAnalyzerSubagent {
        +pdf_parser: PDFParser
        +ppt_parser: PPTParser
        +execute(parameters)
    }

    class VideoAnalyzerSubagent {
        +frame_extractor: FrameExtractor
        +audio_transcriber: AudioTranscriber
        +execute(parameters)
    }

    class ImageAnalyzerSubagent {
        +ocr_tool: OCRTool
        +captioning_tool: CaptioningTool
        +execute(parameters)
    }

    Subagent <|-- SimpleSubagent
    SimpleSubagent <|-- QueryRewriterSubagent
    SimpleSubagent <|-- DocumentAnalyzerSubagent
    SimpleSubagent <|-- VideoAnalyzerSubagent
    SimpleSubagent <|-- ImageAnalyzerSubagent
```

---

## 7. 工具类图

```mermaid
classDiagram
    class Tool {
        <<abstract>>
        +name: str
        +description: str
        +config: dict
        +execute(kwargs)* ToolResult
    }

    class SubagentManagerTool {
        +name: "spawn_subagent"
        +subagent_factory: SubagentFactory
        +execute(task_type, parameters)
    }

    class VectorSearchTool {
        +name: "vector_search"
        +vector_store: VectorStore
        +execute(query, filters)
    }

    class EmbeddingGeneratorTool {
        +name: "generate_embedding"
        +embedding_provider: EmbeddingProvider
        +execute(content, modality)
    }

    class ImageOCRTool {
        +name: "extract_text_from_image"
        +ocr_provider: OCRProvider
        +execute(image_path)
    }

    class ImageCaptioningTool {
        +name: "caption_image"
        +vision_provider: VisionProvider
        +execute(image_path)
    }

    Tool <|-- SubagentManagerTool
    Tool <|-- VectorSearchTool
    Tool <|-- EmbeddingGeneratorTool
    Tool <|-- ImageOCRTool
    Tool <|-- ImageCaptioningTool
```

---

## 8. 提供者类图

```mermaid
classDiagram
    class LLMProvider {
        <<abstract>>
        +model: str
        +api_key: str
        +chat(messages, tools)* ChatResponse
    }

    class VisionProvider {
        <<abstract>>
        +model: str
        +understand_image(image, prompt)* str
    }

    class OCRProvider {
        <<abstract>>
        +extract_text(image_path)* str
    }

    class EmbeddingProvider {
        <<abstract>>
        +embed_text(text)* List[float]
        +embed_image(image)* List[float]
    }

    class OpenAILLM {
        +chat(messages, tools) ChatResponse
    }

    class OpenAIVision {
        +understand_image(image, prompt) str
    }

    class TesseractOCR {
        +extract_text(image_path) str
    }

    class CLIPEmbedding {
        +embed_text(text) List[float]
        +embed_image(image) List[float]
    }

    LLMProvider <|-- OpenAILLM
    VisionProvider <|-- OpenAIVision
    OCRProvider <|-- TesseractOCR
    EmbeddingProvider <|-- CLIPEmbedding
```

---

## 9. 部署架构

```mermaid
graph TB
    subgraph "客户端层"
        Browser[Web Browser]
        CLI[CLI Client]
        APIUser[API User]
    end

    subgraph "负载均衡层"
        Nginx[Nginx<br/>Load Balancer]
    end

    subgraph "应用层"
        App1[MM-RAG Agent 1]
        App2[MM-RAG Agent 2]
        App3[MM-RAG Agent N]
    end

    subgraph "模型服务层"
        OpenAI[OpenAI API]
        HF[HuggingFace<br/>Inference]
        Local[Local Models<br/>vLLM/Ollama]
    end

    subgraph "存储层"
        Redis[Redis<br/>Cache/Session]
        VectorDB[Vector DB<br/>ChromaDB/Milvus]
        S3[S3/MinIO<br/>File Storage]
        PG[PostgreSQL<br/>Metadata]
    end

    Browser --> Nginx
    CLI --> Nginx
    APIUser --> Nginx

    Nginx --> App1
    Nginx --> App2
    Nginx --> App3

    App1 --> OpenAI
    App2 --> OpenAI
    App3 --> OpenAI

    App1 --> HF
    App2 --> HF
    App3 --> HF

    App1 --> Local
    App2 --> Local
    App3 --> Local

    App1 --> Redis
    App2 --> Redis
    App3 --> Redis

    App1 --> VectorDB
    App2 --> VectorDB
    App3 --> VectorDB

    App1 --> S3
    App2 --> S3
    App3 --> S3

    App1 --> PG
    App2 --> PG
    App3 --> PG
```

---

## 使用说明

以上 Mermaid 图表可以在以下平台渲染：

1. **GitHub** - 直接在 README.md 中显示
2. **VS Code** - 安装 "Markdown Preview Mermaid Support" 插件
3. **Typora** - 原生支持 Mermaid
4. **在线编辑器** - https://mermaid.live/
