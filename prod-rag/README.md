# Production RAG Application for Human Nutrition

A production-ready Retrieval-Augmented Generation (RAG) chatbot built from scratch, specializing in human nutrition knowledge. This application processes a comprehensive nutrition PDF and enables intelligent Q&A through semantic search and large language model integration.

## 🏗️ Architecture Overview

<img src = 'images/simple-local-rag-workflow-flowchart.png' alt = "simple-local-rag-workflow-flowchart">

This RAG system follows a modern, scalable architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG Application Stack                    │
├─────────────────────────────────────────────────────────────┤
│  Frontend (Next.js 15 + React 19)                           │
│  ├── Chat Interface (page.tsx)                              │
│  ├── TailwindCSS Styling                                    │
│  └── Real-time Response Streaming                           │
├─────────────────────────────────────────────────────────────┤
│  Backend API (Next.js API Routes)                           │
│  ├── /api/chat - Main RAG endpoint (route.ts)               │
│  ├── Query Embedding (OpenAI text-embedding-3-small)        │
│  ├── Semantic Search (Supabase Vector DB)                   │
│  └── Response Generation (GPT-4o-mini)                      │
├─────────────────────────────────────────────────────────────┤
│  Data Processing Pipeline (Python)                          │
│  ├── PDF Text Extraction (PyMuPDF)                          │
│  ├── Intelligent Chunking (Sentence-based)                  │
│  ├── Embedding Generation (OpenAI API)                      │
│  └── Vector Storage (Supabase/pgvector)                     │
├─────────────────────────────────────────────────────────────┤
│  Database & Vector Store                                    │
│  ├── Supabase (PostgreSQL + pgvector)                       │
│  ├── Vector Similarity Search                               │
│  └── Metadata Storage (page numbers, sources)               │
└─────────────────────────────────────────────────────────────┘
```

## 📊 Core Components

### 1. Data Ingestion Pipeline (`ingest.py`)

**Purpose**: Processes the human nutrition PDF and creates searchable embeddings

**Key Features**:
- **Smart Text Extraction**: Uses PyMuPDF for high-quality PDF text extraction
- **Advanced Chunking**: Sentence-based chunking with configurable overlap (20 sentences per chunk, 2 sentence overlap)
- **Token Management**: Enforces token limits (1300 max, 50 min) for optimal embedding quality
- **Text Cleaning**: Handles hyphenation across line breaks and normalizes whitespace
- **Batch Processing**: Efficient batch embedding generation (100 texts per batch)
- **Metadata Preservation**: Tracks page numbers and source information

**Configuration**:
```python
SENTS_PER_CHUNK = 20    # Sentences per chunk
SENT_OVERLAP = 2        # Overlap between chunks
MAX_TOKENS = 1300       # Maximum tokens per chunk
MIN_TOKENS = 50         # Minimum tokens per chunk
EMBED_MODEL = "text-embedding-3-small"  # 1536 dimensions
```

### 2. Vector Database (Supabase + pgvector)

**Schema Design**:
```sql
CREATE TABLE chunks (
  id BIGSERIAL PRIMARY KEY,
  doc_id TEXT NOT NULL,
  chunk_index INT NOT NULL,
  content TEXT NOT NULL,
  metadata JSONB DEFAULT '{}'::jsonb,
  embedding VECTOR(1536)
);
```

**Similarity Search Function**:
- Implements cosine similarity search with optional metadata filtering
- Optimized with IVFFlat index for fast approximate nearest neighbor search
- Returns ranked results with similarity scores

### 3. Chat API (`/api/chat/route.ts`)

**RAG Pipeline Implementation**:

1. **Query Embedding**: Converts user questions to 1536-dimensional vectors
2. **Semantic Retrieval**: Searches top 8 most relevant chunks using cosine similarity
3. **Context Assembly**: Constructs context with page number citations
4. **Response Generation**: Uses GPT-4o-mini with strict RAG instructions
5. **Source Attribution**: Returns both answer and source references

**Key Features**:
- Server-side only (service role key protection)
- Dynamic routing (no caching for fresh responses)
- Error handling and validation
- Temperature control (0.2) for consistent responses

### 4. Frontend Interface (`page.tsx`)

**Chat Experience**:
- Real-time conversation interface
- Message history management
- Source citation display with similarity scores
- Loading states and error handling
- Responsive design with TailwindCSS

## 🚀 Getting Started

### Prerequisites

- Node.js 18+ and npm/yarn
- Python 3.8+ with pip
- Supabase account
- OpenAI API key

### 1. Environment Setup

Create `.env` files with your API keys:

**Root `.env`:**
```bash
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
OPENAI_API_KEY=your_openai_api_key
```

**Frontend `.env.local`:**
```bash
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
OPENAI_API_KEY=your_openai_api_key
```

### 2. Database Setup

Run the SQL commands from `PROMPT.md` in your Supabase SQL Editor:

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create chunks table
CREATE TABLE IF NOT EXISTS public.chunks (
  id bigserial primary key,
  doc_id text not null,
  chunk_index int not null,
  content text not null,
  metadata jsonb default '{}'::jsonb,
  embedding vector(1536)
);

-- Create index for fast similarity search
CREATE INDEX IF NOT EXISTS idx_chunks_embedding
  ON public.chunks USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100);

-- Create similarity search function
CREATE OR REPLACE FUNCTION public.match_documents(
  query_embedding vector(1536),
  match_count int default 5,
  filter jsonb default '{}'::jsonb
)
RETURNS TABLE (
  id bigint,
  doc_id text,
  chunk_index int,
  content text,
  metadata jsonb,
  similarity float
)
LANGUAGE plpgsql STABLE
AS $$
BEGIN
  RETURN QUERY
  SELECT
    c.id,
    c.doc_id,
    c.chunk_index,
    c.content,
    c.metadata,
    1 - (c.embedding <=> query_embedding) as similarity
  FROM public.chunks c
  WHERE (filter = '{}'::jsonb) OR (c.metadata @> filter)
  ORDER BY c.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;
```

### 3. Data Ingestion

Install Python dependencies and run the ingestion script:

```bash
# Install Python packages
pip install pymupdf tiktoken supabase openai tqdm python-dotenv

# Process the nutrition PDF
python ingest.py
```

This will:
- Extract text from `human-nutrition-text.pdf`
- Create semantic chunks with overlap
- Generate embeddings using OpenAI
- Store everything in Supabase

### 4. Frontend Setup

Navigate to the chat application and start the development server:

```bash
cd rag-chat
npm install
npm run dev
```

The application will be available at `http://localhost:3000`

## 🔧 Technical Details

### Chunking Strategy

The application uses an intelligent sentence-based chunking approach:

- **Sentence Splitting**: Uses regex pattern `(?<=[.!?])\s+` for natural breaks
- **Overlap Strategy**: 2-sentence overlap between chunks prevents information loss
- **Token Awareness**: Monitors token count to stay within model limits
- **Quality Control**: Filters out very short fragments (< 50 tokens)

### Embedding Model

- **Model**: `text-embedding-3-small` (1536 dimensions)
- **Advantages**: Cost-effective, fast inference, good quality for domain-specific content
- **Tokenizer**: Uses `cl100k_base` encoding for accurate token counting

### Search Performance

- **Index Type**: IVFFlat with 100 lists for approximate nearest neighbor search
- **Distance Metric**: Cosine similarity (optimal for normalized embeddings)
- **Query Time**: Sub-second response times for typical queries
- **Accuracy**: High recall with configurable result count

### LLM Integration

- **Model**: GPT-4o-mini for cost-effective, high-quality responses
- **Temperature**: 0.2 for consistent, factual answers
- **System Prompt**: Strict RAG instructions to prevent hallucination
- **Citation**: Automatic source attribution with page numbers

## 📁 Project Structure

```
prod-rag/
├── .env                              # Environment variables
├── human-nutrition-text.pdf          # Source document (26MB)
├── ingest.py                        # Data processing pipeline
├── test_embeddings.py               # Embedding testing script  
├── PROMPT.md                        # SQL setup and development notes
└── rag-chat/                       # Next.js frontend application
    ├── .env.local                   # Frontend environment variables
    ├── package.json                 # Dependencies and scripts
    ├── src/
    │   └── app/
    │       ├── globals.css          # Global styles
    │       ├── layout.tsx           # Root layout component
    │       ├── page.tsx             # Main chat interface
    │       └── api/
    │           └── chat/
    │               └── route.ts     # RAG API endpoint
    └── [Next.js build/config files]
```

## 🛠️ Dependencies

### Python Dependencies
- `pymupdf`: PDF text extraction
- `tiktoken`: OpenAI tokenization
- `supabase`: Database client
- `openai`: Embedding and chat completion APIs
- `tqdm`: Progress bars
- `python-dotenv`: Environment variable management

### Node.js Dependencies
- `next`: React framework (v15.5.4)
- `react`: UI library (v19.1.0)
- `@supabase/supabase-js`: Supabase client
- `openai`: OpenAI API client
- `tailwindcss`: Utility-first CSS framework
- `typescript`: Type safety

## ⚡ Performance Characteristics

- **Ingestion Speed**: ~100 chunks embedded per batch
- **Search Latency**: < 500ms for similarity search
- **Embedding Dimension**: 1536 (optimized for quality/speed balance)
- **Chunk Size**: 20 sentences (~400-800 tokens typically)
- **Retrieval Count**: 8 chunks per query (configurable)
- **Model Response**: GPT-4o-mini with 0.2 temperature

## 🎯 Use Cases

This RAG system is specifically designed for:

- **Nutrition Consultation**: Evidence-based answers about dietary guidelines
- **Research Assistance**: Quick access to specific nutritional information
- **Educational Support**: Learning about vitamins, minerals, and dietary requirements
- **Professional Reference**: Healthcare and nutrition professionals

## 🔒 Security Features

- **API Key Protection**: Server-side only OpenAI and Supabase keys
- **Service Role**: Database access through service role (not anon key)
- **Environment Isolation**: Separate environment files for different components
- **No Client Exposure**: Sensitive credentials never reach the browser

## 📈 Scalability Considerations

- **Horizontal Scaling**: Stateless API design supports multiple instances
- **Vector Index**: IVFFlat index scales to millions of vectors
- **Batch Processing**: Configurable batch sizes for optimal throughput
- **Caching Opportunities**: Response caching can be added for repeated queries
- **Model Swapping**: Easy to switch between embedding models or LLMs

## 🧪 Testing

The project includes `test_embeddings.py` for validating the search functionality with sample queries:

```python
queries = [
    "How often should infants be breastfed?",
    "What are symptoms of pellagra?", 
    "How does saliva help with digestion?",
    "What is the RDI for protein per day?",
    "water soluble vitamins",
    "What are micronutrients?"
]
```

Run with: `python test_embeddings.py`

## 📝 Development Notes

The `PROMPT.md` file contains detailed development context including:
- SQL schema definitions
- Complete code examples
- Frontend design requirements  
- Supabase configuration steps

This serves as both documentation and development reference for the entire system.

---

**Built by**: Manoj Chandrashekar  
**Referenced Repo**:https://github.com/mrdbourke/simple-local-rag/tree/main  
**Workshop followed**: Vizuara AI  
**Vizuara Labs**: https://home.vizuara.ai/  
**Architecture**: Production-ready RAG with Next.js, Supabase, and OpenAI  
**Domain**: Human Nutrition Knowledge Base