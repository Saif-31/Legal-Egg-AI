# Legal Egg AI 🥚 - Serbian Supreme Court Criminal Practice Assistant 

An AI-powered legal assistant specializing in Serbian Supreme Court criminal practice patterns and precedents.

## Features

- **Comprehensive Legal Analysis**: Detailed analysis of criminal law cases and precedents
- **Bilingual Processing**: Handles queries in English and Serbian
- **Structured Responses**: Organized format following legal documentation standards
- **Real-time Updates**: Incorporates latest court decisions and practice changes
- **Practice-oriented**: Focuses on practical application of legal principles

## Technical Architecture

### Core Components

1. **Frontend (Streamlit)**
   - Clean, intuitive chat interface
   - Example query suggestions
   - Session management
   - Response formatting

2. **Backend**
   - LangChain for orchestration
   - Mistral AI for language processing
   - Pinecone vector database
   - HuggingFace embeddings

3. **Data Processing**
   - Query refinement pipeline
   - Context retrieval system
   - Response structuring
   - Bilingual processing

## Installation

1. Clone the repository:
   ```bash
   git clone [repository-url]
   cd Court-Practices
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage

The assistant provides:
- Supreme Court case analysis
- Legal precedent retrieval
- Procedural guidance
- Practice pattern analysis
- Strategic legal insights

### Query Examples
- Court positions on specific crimes
- Evidence standards analysis
- Procedural requirement details
- Sentencing pattern analysis
- Defense strategy insights

## Development

### Project Structure
```
Court-Practices/
├── app.py              # Main application
├── requirements.txt    # Dependencies
├── .env               # Configuration
├── GUIDE.md           # User guide
└── OVERVIEW.md        # Technical overview
```

[![Watch the video](https://img.youtube.com/vi/b9bNDEo3HgE/maxresdefault.jpg)](https://youtu.be/b9bNDEo3HgE)
