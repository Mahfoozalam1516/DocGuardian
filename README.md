# 📚 Enhanced Document Q&A Chatbot

A Streamlit-based chatbot that allows users to upload documents (PDF, DOCX, TXT) and ask questions about their content using the HuggingFace API for natural language processing.

## Features

- 📝 Support for multiple document formats (PDF, DOCX, TXT)
- 🔑 Manual API key configuration within the app
- 💬 Interactive chat interface
- 📊 Real-time statistics and usage metrics
- 📥 Downloadable chat history
- 🔍 Source document references for answers
- 📈 Visual analytics of chat usage

## Prerequisites

- Python 3.8 or higher
- HuggingFace API key ([Get one here](https://huggingface.co/settings/tokens))

## Installation

1. Clone this repository:

```bash
git clone <repository-url>
cd document-qa-chatbot
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:

```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (usually `http://localhost:8501`)

3. Enter your HuggingFace API key in the sidebar

4. Upload your documents and click "Process Documents"

5. Start asking questions about your documents!

## Features in Detail

### Document Processing

- Supports PDF, DOCX, and TXT files
- Automatic text extraction and chunking
- Document statistics tracking

### Chat Interface

- Real-time question answering
- Source document references
- Chat history with download option
- Response time tracking

### Analytics

- Question count metrics
- Average response time
- Document processing statistics
- Usage visualization

## Technical Details

The application uses:

- Streamlit for the web interface
- LangChain for document processing and chat functionality
- HuggingFace's language models for text generation
- FAISS for vector storage and similarity search
- Sentence Transformers for text embeddings

## Security

- API keys are stored securely in session state
- No data is permanently stored on the server
- All processing is done locally

## Limitations

- Maximum file size depends on available memory
- Processing time varies with document size
- API rate limits apply based on your HuggingFace account type

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
