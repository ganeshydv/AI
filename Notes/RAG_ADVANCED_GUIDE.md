# Advanced RAG Guide (PDF with text, tables, images, charts, formulas)

## 0. Naive vs Advanced RAG

| Stage | Naive RAG | Advanced RAG |
| --- | --- | --- |
| Ingestion | Plain text extraction, fixed-size chunks | Layout-aware parsing, OCR, per-type enrichment (VLM/LLM), section-based chunking, metadata |
| Query | Raw query embedded | Rewrite with chat history, filter extraction |
| Retrieval | Dense vector top-k | Hybrid (dense + BM25 sparse) with metadata filters, RRF fusion |
| Post-retrieval | Chunks go straight into the prompt | Cross-encoder rerank, parent/neighbour expansion, fetch original image/table |
| Generation | Text LLM | Multimodal LLM when needed, citations |

## 1. Core concepts

### OCR vs VLM

| Tool | Answers | Good for | Not good for |
| --- | --- | --- | --- |
| **OCR** (PP-OCR) | "What text is written in this image?" | Scanned pages, screenshots, labels and numbers on charts | Describing meaning (trends, diagrams, photos) |
| **VLM** (GPT-4o, Claude, Qwen-VL) | "What does this image show or mean?" | Chart/diagram/photo summaries | Exact character-level text |

Plain PP-OCR returns only text boxes. **PP-StructureV3** (PaddleOCR layout pipeline) adds layout detection, table structure and formula recognition. Charts and diagrams still need a VLM.

### How to know a block is an image, table or formula

There is no stream that announces "an image starts here". Detection happens per page, before chunking:

1. **Native PDF objects (digital PDFs):** PyMuPDF `page.get_images()`, `page.get_image_bbox()`, `page.find_tables()` return embedded images and tables with positions.
2. **Layout detection (scanned PDFs, vector charts):** render the page to an image, then a layout model (PP-StructureV3, Docling, MinerU) returns labeled boxes (`text`, `title`, `table`, `figure`, `formula`) with coordinates. Vector charts are drawn with path commands, not embedded images, so only this method finds them.

### Dense vs sparse vs cross-encoder

| | Dense (bi-encoder) | Sparse (BM25) | Cross-encoder |
| --- | --- | --- | --- |
| Input | Query and chunk encoded separately | Query terms vs chunk terms | Query and chunk together |
| Strength | Meaning, paraphrases | Exact words, IDs, numbers, codes | Best relevance judgement |
| Speed | Fast (ANN/HNSW index) | Fast (inverted index) | Slow, so only top 20-50 |
| Role | Recall | Recall | Precision |

## 2. Ingestion pipeline (offline, runs once per PDF)

Order matters: parse the whole page into typed blocks first, enrich each block, then chunk, then embed. Chunking never pauses mid-way to call a VLM.

```mermaid
flowchart TD
    S0[0. Setup: create Qdrant collection + payload indexes] --> A
    A[1. PDF upload] --> B{2. Page has selectable text?}
    B -- yes digital --> C[Native parse: PyMuPDF / Docling]
    B -- no scanned --> D[Render page to image]
    C --> E[3. Layout detection per page]
    D --> E
    E --> F[4. Typed blocks: type + page + bbox + reading order]
    F --> T[Text / title]
    F --> TB[Table]
    F --> FM[Formula]
    F --> IM[Figure / chart]
    T --> T1[5a. Native text, or PP-OCR if scanned]
    TB --> TB1[5b. Table to Markdown/HTML, LLM summary]
    FM --> FM1[5c. Formula OCR to LaTeX, LLM explanation]
    IM --> IM1[5d. Crop, save to S3, PP-OCR labels, VLM caption]
    T1 & TB1 & FM1 & IM1 --> CH[6. Chunking: one chunk per block, split if long, add section title]
    CH --> DE[7a. Dense embedding model to float vector]
    CH --> SP[7b. Sparse encoder BM25 / SPLADE to term:weight vector]
    DE & SP --> Q[(8. Upsert to Qdrant: dense + sparse + payload)]
    IM1 -. asset_url .-> S3[(S3: original images, raw tables)]
    TB1 -. table_id .-> S3
```

#### Parser stage detail (Docling)

The parser's job is to split the PDF into typed blocks and to know which section each block belongs to. Docling does this in one pass and returns a `DoclingDocument`.

```text
[ Input PDF ]  (digital text, scanned pages, handwriting, tables, formulas, charts, flowcharts)
      │
      ▼
[ Docling Parser ]
  - layout model: labels every region (section_header, text, list_item, table,
    picture, formula, code, caption, page_header/footer)
  - reading order + section hierarchy (which heading owns which block)
  - TableFormer: table cells to rows/columns
  - OCR: only for scanned pages and image regions (EasyOCR / Tesseract / RapidOCR)
      │
      ▼
[ DoclingDocument ]  typed items, each with: type, page, bbox, reading order, parent section
      │
      ├── section_header / text / list_item ─> Markdown text (OCR text if scanned)
      ├── table ─────────────────────────────> Markdown/HTML + raw JSON (S3) + LLM summary
      ├── formula ───────────────────────────> LaTeX + LLM explanation
      ├── picture (chart, flowchart) ────────> crop -> S3 -> VLM caption + OCR labels
      └── handwriting / low-confidence OCR ──> crop -> multimodal VLM transcription
      │
      ▼
[ Chunking ]  one chunk per item, section title + parent ids attached (Docling HybridChunker)
      │
      ▼
[ Embedding: dense + sparse ] ─> [ Qdrant ]  (+ S3 for crops and raw tables)

Query time is a separate flow (section 4):
Qdrant retrieval -> rerank -> multimodal LLM gets text + the original image clips
```

Notes on this stage:

- **Section knowledge comes from the parser.** `section_header` items and the hierarchy tell you which heading each block sits under. Store it as the `section` payload field and prepend it to chunk text.
- **Handwriting is not a block type.** The layout model only labels regions as text or picture. Standard OCR is weak on handwriting, so route by policy: low OCR confidence, or a region flagged as handwritten, goes to the VLM.
- **Tables and formulas are separate branches.** They need their own handling (structure, LaTeX), not the text path.
- **The final multimodal VLM runs at query time**, not during ingestion. Ingestion VLM calls only produce captions and transcripts.

### Step by step

| # | Step | What happens |
| --- | --- | --- |
| 0 | Setup | Create the collection with named vectors (`dense`, `sparse` with `modifier: idf`). Create payload indexes on filter fields (`doc_id`, `type`, `page`, `year`, ...) before loading data. |
| 1 | Upload | Receive the PDF. Small: process synchronously. Large: queue (SQS/Celery), process page by page with parallel workers. |
| 2 | Classify page | Selectable text means native parse. No text layer means render to image and use OCR. |
| 3 | Layout detection | Label regions on each page: text, title, table, figure, formula, header/footer. |
| 4 | Typed blocks | Each block has `type`, `page`, `bbox`, `reading_order`, `section`. |
| 5 | Per-type extraction/enrichment | See table below. VLM and LLM calls run as parallel or queued jobs, cached by image hash. |
| 6 | Chunking | One chunk per block. Long text is split by section/paragraph. Attach section heading and neighbour ids (parent-child). Do not merge a table or figure into surrounding text. |
| 7 | Embedding | Dense vector from the embedding model, sparse vector from BM25 (or SPLADE) over the same chunk text. |
| 8 | Store | Upsert one Qdrant point per chunk. Upload assets to S3 and link them in the payload. |

### Per-type handling

| Block type | Extraction | Enrichment | Text that gets embedded | Stored outside the vector DB |
| --- | --- | --- | --- | --- |
| Text | Native text, or PP-OCR | None | The chunk itself | Nothing |
| Table | Table structure to Markdown/HTML | LLM summary for large tables | Markdown plus summary | Raw table JSON/HTML |
| Image/chart | Crop, PP-OCR for labels | VLM caption plus data summary, given nearby caption and section title as context | Caption plus OCR labels | Cropped PNG in S3 |
| Formula | Formula OCR to LaTeX | LLM plain-language explanation | LaTeX plus explanation | None |

Images are never embedded as pixels in this approach. Their VLM summary is.

## 3. What is stored in Qdrant

Both vectors live on the same point as named vectors. Sparse is computed at index time, not only at search time.

```json
{
  "id": "b3f1c0de-5a7e-5c1a-9d2b-3e8f6a1c4d70",
  "vectors": {
    "dense":  [0.012, -0.44],
    "sparse": { "indices": [1021, 5530, 90412], "values": [1.8, 0.9, 2.4] }
  },
  "payload": {
    "chunk_id": "doc12_p7_fig2",
    "doc_id": "doc12",
    "page": 7,
    "type": "image",
    "bbox": [72, 300, 520, 640],
    "section": "3.2 Financial Results",
    "text": "Bar chart: quarterly revenue Q1-Q4 2024, 2.1M to 3.4M",
    "asset_url": "s3://bucket/doc12/p7_fig2.png",
    "raw_table": null
  }
}
```

Embed the searchable text. Keep the raw table or image in the payload or S3.

Qdrant point ids must be an unsigned integer or a UUID. Generate a deterministic UUID (for example `uuid5(NAMESPACE, "doc12_p7_fig2")`) so re-ingesting the same chunk overwrites it, and keep the readable id in the payload as `chunk_id`.

### How sparse (BM25) works

Dense vectors capture meaning ("car" is near "automobile"). Sparse vectors capture exact terms: names, IDs, numbers, error codes.

At index time:

1. Tokenize the chunk text (lowercase, optional stemming and stop-word removal).
2. Map each unique word to an integer id. The vector has one dimension per vocabulary word and is mostly zeros, hence "sparse".
3. Compute a weight per word with BM25:

$$
\text{score}(q,d)=\sum_{t\in q}\text{IDF}(t)\cdot\frac{tf(t,d)\,(k_1+1)}{tf(t,d)+k_1\left(1-b+b\frac{|d|}{\text{avgdl}}\right)}
$$

- $tf$: how often the term appears in the chunk.
- $\text{IDF}$: rarity of the term across all chunks. Rare terms weigh more.
- $|d|/\text{avgdl}$: length normalization, so long chunks are not favoured.
- Typical constants: $k_1\approx1.2$, $b\approx0.75$.

4. Store only the non-zero entries as `{indices, values}`.

At query time:

- The query is tokenized the same way. Only which terms are present matters.
- Qdrant takes the dot product between the query's sparse vector and the stored ones. With `modifier: idf` on the sparse vector, Qdrant maintains IDF on the server as data changes. FastEmbed's `Qdrant/bm25` model produces the vectors.
- Alternative: learned sparse (SPLADE, BGE-M3 sparse). A neural model produces the weights and expands terms ("car" also activates "vehicle"). More accurate, but heavier.

## 4. Query pipeline (online, per user question)

```mermaid
flowchart TD
    U[1. User question] --> R[2. LLM rewrite with chat history to standalone query]
    R --> M[3. Optional: extract metadata filters, e.g. doc, year, type]
    R --> DE[4a. Dense embed query]
    R --> SE[4b. Sparse encode query]
    M --> P[Payload filter]
    P --> DS
    P --> SS
    DE --> DS[5a. Dense search, top 20-50, filtered]
    SE --> SS[5b. Sparse BM25 search, top 20-50, filtered]
    DS & SS --> RRF[6. RRF fusion]
    RRF --> C[Top 20-30 candidates]
    C --> X[7. Cross-encoder rerank: query + chunk text]
    X --> T5[Top 5]
    T5 --> EX[8. Expand: parent section, neighbours, fetch S3 image or raw table]
    EX --> L[9. LLM answer with citations, multimodal if image/chart]
```

| # | Step | What happens |
| --- | --- | --- |
| 1-2 | Rewrite | The LLM uses previous questions and answers to turn a follow-up ("what about Q4?") into a standalone query. |
| 3 | Filters (optional) | The LLM extracts metadata filters from the question ("in the 2024 report" becomes `year = 2024`). |
| 4 | Encode | Dense embedding and sparse encoding of the rewritten query. |
| 5 | Search | Dense and sparse searches run in parallel, each with the same filter, each returning its own top N. |
| 6 | Fusion | RRF merges the two ranked lists. |
| 7 | Rerank | A cross-encoder scores each (query, chunk) pair and keeps the top 5. It reads text only, so images are ranked by caption and OCR text. |
| 8 | Expand | Fetch parent section and neighbouring chunks. For `type=table` fetch the raw table. For `type=image` fetch the S3 image. |
| 9 | Generate | LLM answers from the context and cites doc and page. A multimodal LLM receives the actual image for chart questions. |

### Hybrid query in Qdrant

Put the filter on each prefetch. A filter only on the outer query does not constrain the prefetch candidates.

```python
client.query_points(
    collection_name="docs",
    prefetch=[
        Prefetch(query=dense_vec, using="dense", limit=30, filter=flt),
        Prefetch(query=sparse_vec, using="sparse", limit=30, filter=flt),
    ],
    query=FusionQuery(fusion=Fusion.RRF),
    limit=20,
)
```

### How filtering works inside the search

The filter is not applied after the search. It is applied during the search, so the top N is taken from points that already match.

| Filter selectivity | What Qdrant does |
| --- | --- |
| Very selective (few matches) | The payload index gives the matching point IDs, then only those points are scored (exact). |
| Broad (many matches) | HNSW graph traversal checks the filter on each visited point and skips non-matching ones until N valid results are found (filterable HNSW). |

Qdrant picks the strategy per query using a cardinality estimate from the payload index. For sparse search, the inverted index finds candidates and each one is checked against the filter before it counts.

Post-filtering is avoided because it can leave fewer than N results, or none. The payload index does not rank anything. It only makes filtering fast and feeds the planner. Create payload indexes on filter fields before inserting data.

### RRF (Reciprocal Rank Fusion)

Dense scores (cosine, e.g. 0.83) and BM25 scores (e.g. 14.2) are on different scales, so they cannot be added. RRF uses ranks only:

$$
\text{RRF}(d)=\sum_{r\in\{\text{dense},\text{sparse}\}}\frac{1}{k+\text{rank}_r(d)},\quad k=60
$$

| Chunk | Dense rank | Sparse rank | RRF score |
| --- | --- | --- | --- |
| A | 1 | 5 | 1/61 + 1/65 = 0.0318 |
| B | 3 | 1 | 1/63 + 1/61 = 0.0323 |
| C | 2 | not found | 1/62 = 0.0161 |

B wins because both retrievers rank it well.

## 5. API walkthrough (Qdrant REST, in flow order)

Base URL `http://localhost:6333`. Add the header `api-key: <key>` if auth is enabled. Collection name: `docs`.

### 5.1 Ingestion APIs (in order)

| Order | Call | Purpose |
| --- | --- | --- |
| 1 | `PUT /collections/docs` | Create collection with dense and sparse vector slots |
| 2 | `PUT /collections/docs/index` | Create payload indexes for filter fields |
| 3 | Parse PDF (Docling) | Typed blocks, no vector DB call |
| 4 | VLM / LLM call | Caption images, summarize tables, explain formulas |
| 5 | Dense embedding call | Chunk text to float vector |
| 6 | BM25 sparse encoding | Chunk text to `{indices, values}` |
| 7 | Upload crop/raw table to S3 | Get `asset_url` |
| 8 | `PUT /collections/docs/points` | Upsert chunks with both vectors and payload |

**Step 1. Create the collection**

`PUT /collections/docs`

```json
{
  "vectors": {
    "dense": { "size": 1024, "distance": "Cosine" }
  },
  "sparse_vectors": {
    "sparse": { "modifier": "idf" }
  }
}
```

`size` must equal the output dimension of your dense embedding model (1024 for bge-m3). `modifier: idf` makes Qdrant compute the IDF part of BM25 on the server.

**Step 2. Create payload indexes (before loading data)**

`PUT /collections/docs/index?wait=true`, one call per field:

```json
{ "field_name": "doc_id", "field_schema": "keyword" }
```
```json
{ "field_name": "type", "field_schema": "keyword" }
```
```json
{ "field_name": "page", "field_schema": "integer" }
```
```json
{ "field_name": "year", "field_schema": "integer" }
```

**Step 3. Parse the PDF with Docling** (no Qdrant call)

What Docling gives you in this step:

| Capability | What it does | Why it matters for RAG |
| --- | --- | --- |
| Layout analysis | Labels each region: `section_header`, `text`, `list_item`, `table`, `picture`, `formula`, `code`, `caption`, `page_header`, `page_footer` | You know the block type without guessing, so each type gets its own handling |
| Reading order | Orders blocks as a human would read them, including multi-column pages | Chunks are in the correct sequence |
| Section hierarchy | Links each block to its parent heading | Fills the `section` payload and lets you prepend headings to chunks |
| Table structure (TableFormer) | Recovers rows, columns, merged cells and headers | Tables can be exported as Markdown, HTML or a DataFrame instead of garbled text |
| OCR | Runs EasyOCR, Tesseract or RapidOCR on scanned pages and image regions. Set `force_full_page_ocr` for fully scanned PDFs | One pipeline for digital and scanned pages |
| Formula enrichment | Converts formula regions to LaTeX (`do_formula_enrichment`) | Searchable and readable formulas |
| Picture handling | Crops each picture (`generate_picture_images`), links its caption, and can classify it (chart, diagram, logo) | Feeds the VLM step and skips decorative images |
| Provenance | Every item has `page_no` and `bbox` | Page citations and linking chunks back to the PDF |
| Structure-aware chunking | `HybridChunker` splits by document structure and token limit, and keeps headings in chunk metadata | Better chunks than fixed-size splitting |
| Export formats | Markdown, HTML, lossless JSON (`DoclingDocument`) | Store the JSON to re-chunk later without re-parsing |

What it does not do: it does not understand charts (a VLM does that), and it does not reliably read handwriting.

```python
from docling.document_converter import DocumentConverter, PdfFormatOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.chunking import HybridChunker

opts = PdfPipelineOptions()
opts.do_ocr = True                     # OCR on scanned pages and image regions
opts.do_table_structure = True         # TableFormer
opts.do_formula_enrichment = True      # formulas to LaTeX
opts.generate_picture_images = True    # keep crops of figures
opts.images_scale = 2.0                # sharper crops for the VLM

converter = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOptions(pipeline_options=opts)}
)
doc = converter.convert("doc12.pdf").document

# Walk typed items in reading order
for item, level in doc.iterate_items():
    page = item.prov[0].page_no
    bbox = item.prov[0].bbox
    label = item.label.value                 # "table", "picture", "text", "formula", ...
    if label == "table":
        md = item.export_to_markdown(doc=doc)    # also export_to_html / export_to_dataframe
    elif label == "picture":
        img = item.get_image(doc)                # PIL image to upload to S3 and send to the VLM
        caption = item.caption_text(doc)         # nearby caption, useful VLM context
    else:
        text = item.text

# Structure-aware chunks with their headings
chunker = HybridChunker()
for chunk in chunker.chunk(dl_doc=doc):
    headings = chunk.meta.headings               # section path, e.g. ["3 Results", "3.2 Financial Results"]
    text = chunker.contextualize(chunk)          # chunk text with headings prepended
```

Parsed item, as you would keep it before enrichment:

```json
{
  "type": "picture",
  "page": 7,
  "bbox": [72, 300, 520, 640],
  "section": "3.2 Financial Results",
  "caption": "Figure 2. Quarterly revenue",
  "image_path": "crops/doc12_p7_fig2.png"
}
```

Typical use: tables and text use Docling's output directly. Pictures go to S3 and the VLM (step 4). Low-confidence OCR regions and handwriting go to the VLM as well.

**Step 4. VLM caption for a figure** (OpenAI-compatible chat API)

`POST {VLM_URL}/v1/chat/completions`

```json
{
  "model": "gpt-4o",
  "messages": [
    {
      "role": "user",
      "content": [
        { "type": "text", "text": "Section: 3.2 Financial Results. Caption: Figure 2. Describe this chart and list the data values." },
        { "type": "image_url", "image_url": { "url": "data:image/png;base64,<BASE64>" } }
      ]
    }
  ]
}
```

The answer in `choices[0].message.content` becomes the chunk text, together with the PP-OCR labels.

**Steps 5 and 6. Dense and sparse embeddings of the same chunk text**

Dense (OpenAI-compatible embeddings API, for example a self-hosted bge-m3):

`POST {EMBED_URL}/v1/embeddings`

```json
{
  "model": "bge-m3",
  "input": ["Section 3.2 Financial Results. Bar chart: quarterly revenue Q1-Q4 2024, 2.1M to 3.4M"]
}
```

Response: `data[0].embedding` is the float array (1024 values).

Sparse BM25 has no HTTP call in open-source Qdrant. It is computed client-side with FastEmbed:

```python
from fastembed import SparseTextEmbedding

bm25 = SparseTextEmbedding("Qdrant/bm25")
texts = ["Section 3.2 Financial Results. Bar chart: quarterly revenue Q1-Q4 2024, 2.1M to 3.4M"]

# Index side: term frequency weights with length normalization
sp = list(bm25.embed(texts))[0]
sparse_vec = {"indices": sp.indices.tolist(), "values": sp.values.tolist()}
```

Sparse output example: `{"indices": [1021, 5530, 90412], "values": [1.8, 0.9, 2.4]}`. IDF is applied by Qdrant at search time because of `modifier: idf`.

**Step 8. Upsert points**

`PUT /collections/docs/points?wait=true`

```json
{
  "points": [
    {
      "id": "b3f1c0de-5a7e-5c1a-9d2b-3e8f6a1c4d70",
      "vector": {
        "dense": [0.012, -0.44, 0.09],
        "sparse": { "indices": [1021, 5530, 90412], "values": [1.8, 0.9, 2.4] }
      },
      "payload": {
        "chunk_id": "doc12_p7_fig2",
        "doc_id": "doc12",
        "page": 7,
        "year": 2024,
        "type": "image",
        "section": "3.2 Financial Results",
        "text": "Bar chart: quarterly revenue Q1-Q4 2024, 2.1M to 3.4M",
        "bbox": [72, 300, 520, 640],
        "asset_url": "s3://bucket/doc12/p7_fig2.png"
      }
    }
  ]
}
```

Send points in batches (for example 64 to 256 per request). The dense array is shortened here for display.

### 5.2 Query APIs (in order)

| Order | Call | Purpose |
| --- | --- | --- |
| 1 | LLM chat call | Rewrite the question with history, optionally extract filters |
| 2 | Dense embedding call | Query to dense vector |
| 3 | BM25 query encoding | Query to sparse vector |
| 4 | `POST /collections/docs/points/query` | Filtered dense and sparse search with RRF fusion |
| 5 | Reranker call | Cross-encoder scores top candidates |
| 6 | `POST /collections/docs/points/scroll` | Fetch neighbouring chunks |
| 7 | S3 presigned GET | Fetch the original image |
| 8 | LLM chat call | Final answer |

**Step 1. Rewrite and filter extraction**

`POST {LLM_URL}/v1/chat/completions`

```json
{
  "model": "gpt-4o-mini",
  "response_format": { "type": "json_object" },
  "messages": [
    { "role": "system", "content": "Rewrite the last question as a standalone query using the history. Return JSON: {\"query\": string, \"filters\": {\"year\": int|null, \"type\": string|null}}" },
    { "role": "user", "content": "History: Q: Show 2024 revenue. A: Q1 2.1M ... \nQuestion: what about Q4?" }
  ]
}
```

Example result: `{"query": "2024 revenue in Q4", "filters": {"year": 2024, "type": null}}`.

**Steps 2 and 3. Encode the rewritten query**

```json
{ "model": "bge-m3", "input": ["2024 revenue in Q4"] }
```

```python
sp = list(bm25.query_embed("2024 revenue in Q4"))[0]
query_sparse = {"indices": sp.indices.tolist(), "values": sp.values.tolist()}
```

`query_embed` is the query-side function: it keeps which terms are present, without length normalization.

**Step 4. Hybrid search: filtered dense + sparse, fused with RRF**

`POST /collections/docs/points/query`

```json
{
  "prefetch": [
    {
      "query": [0.031, -0.12, 0.27],
      "using": "dense",
      "limit": 30,
      "filter": { "must": [ { "key": "year", "match": { "value": 2024 } } ] }
    },
    {
      "query": { "indices": [1021, 7710], "values": [1.0, 1.0] },
      "using": "sparse",
      "limit": 30,
      "filter": { "must": [ { "key": "year", "match": { "value": 2024 } } ] }
    }
  ],
  "query": { "fusion": "rrf" },
  "limit": 20,
  "with_payload": true
}
```

The filter is repeated inside each prefetch so both searches are constrained. The response lists points with `id`, `score` (the RRF score) and `payload`.

Common filter shapes:

```json
{ "must": [ { "key": "doc_id", "match": { "value": "doc12" } }, { "key": "type", "match": { "any": ["table", "image"] } } ] }
```

```json
{ "must": [ { "key": "page", "range": { "gte": 5, "lte": 9 } } ] }
```

**Step 5. Cross-encoder rerank** (for example a bge-reranker served by Hugging Face TEI)

`POST {RERANK_URL}/rerank`

```json
{
  "query": "2024 revenue in Q4",
  "texts": ["<payload.text of hit 1>", "<payload.text of hit 2>", "<payload.text of hit 20>"]
}
```

Response: `[{"index": 7, "score": 0.97}, {"index": 2, "score": 0.91}, ...]`. Sort by `score` and keep the top 5.

**Step 6. Expand with neighbouring chunks**

`POST /collections/docs/points/scroll`

```json
{
  "filter": {
    "must": [
      { "key": "doc_id", "match": { "value": "doc12" } },
      { "key": "page", "range": { "gte": 6, "lte": 8 } }
    ]
  },
  "limit": 20,
  "with_payload": true,
  "with_vector": false
}
```

To fetch known points by id instead, use `POST /collections/docs/points` with `{"ids": ["<uuid>"], "with_payload": true}`.

**Step 7. Fetch the image.** Generate a presigned S3 GET URL from `payload.asset_url` (SDK call, no Qdrant API), then download it or pass it to the LLM.

**Step 8. Final answer with text and image**

`POST {LLM_URL}/v1/chat/completions`

```json
{
  "model": "gpt-4o",
  "messages": [
    { "role": "system", "content": "Answer only from the context. Cite doc and page." },
    {
      "role": "user",
      "content": [
        { "type": "text", "text": "Context:\n[doc12 p7] Bar chart: quarterly revenue ...\n[doc12 p8] Q4 revenue table ...\n\nQuestion: what about Q4?" },
        { "type": "image_url", "image_url": { "url": "https://<presigned-s3-url>" } }
      ]
    }
  ]
}
```

### 5.3 Whole flow as client code

```python
# INGEST (per chunk batch)
texts = [c["text"] for c in chunks]
dense = embed_dense(texts)                       # POST /v1/embeddings
sparse = list(bm25.embed(texts))                 # FastEmbed Qdrant/bm25
qdrant.upsert("docs", points=[
    PointStruct(id=uuid5(NS, c["chunk_id"]),
                vector={"dense": d, "sparse": SparseVector(indices=s.indices.tolist(), values=s.values.tolist())},
                payload=c["payload"])
    for c, d, s in zip(chunks, dense, sparse)
])

# QUERY
q = rewrite(question, history)                   # LLM call
qd = embed_dense([q["query"]])[0]
qs = list(bm25.query_embed(q["query"]))[0]
hits = qdrant.query_points("docs", prefetch=[
    Prefetch(query=qd, using="dense", limit=30, filter=flt),
    Prefetch(query=SparseVector(indices=qs.indices.tolist(), values=qs.values.tolist()),
             using="sparse", limit=30, filter=flt),
], query=FusionQuery(fusion=Fusion.RRF), limit=20, with_payload=True).points
top5 = rerank(q["query"], hits)[:5]              # POST /rerank
answer = llm_answer(q["query"], expand(top5))    # chat call with text + image
```

## 6. End-to-end summary

```text
INDEX: PDF -> page classify -> layout blocks -> per-type extraction + VLM/LLM enrichment
       -> chunks (one per block + section + parent ids)
       -> dense vector + sparse vector + payload -> Qdrant (+ S3 for assets)

QUERY: question + history -> rewrite -> (filters)
       -> dense top-N || sparse top-N (same filter) -> RRF -> cross-encoder top 5
       -> expand (parents, image, raw table) -> LLM -> answer + citations
```

## 7. Large PDFs and cost

- Process page by page in a queue with parallel workers.
- Use OCR only on pages with no text layer.
- Batch VLM calls and cache by image hash.
- Re-ingest only changed pages (hash per page).

## 8. Alternative: skip OCR with ColPali / ColQwen

These models embed whole page images, so layout, charts and tables are handled in one step. Retrieval is strong for visually complex documents, but storage is heavier and a multi-vector store is needed (Qdrant and Vespa support this). Many teams combine both: parsed text for the main path, ColPali for visually heavy pages.

## 9. Tool choices

- Simplest: Docling or Unstructured, plus a VLM for figures.
- Open-source OCR/layout: PP-StructureV3 or MinerU.
- Managed: LlamaParse or Azure Document Intelligence.
