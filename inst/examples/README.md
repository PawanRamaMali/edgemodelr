# edgemodelr Examples

This directory contains examples for using edgemodelr with modern small language models.

## 📁 Organization

### 🚀 Quick Start Examples
**For immediate solutions and learning:**
- `working_document_analysis.R` - Simple document analysis (works immediately)
- `manual_setup_example.R` - Troubleshooting model downloads
- `document_analysis_example.R` - Alternative analysis approach

### 🎯 Professional Examples  
**For production applications:**
- `01_model_comparison.R` - Modern model selection and comparison
- `02_document_analysis.R` - Advanced document analysis system
- `03_content_generation.R` - Multi-format content generation
- `04_streaming_chat.R` - Interactive streaming conversations
- `05_model_benchmarking.R` - Systematic model evaluation

## 🏃 Quick Start

```r
# Option 1: Quick working solution
source(system.file("examples", "working_document_analysis.R", package = "edgemodelr"))

# Option 2: Professional system  
source(system.file("examples", "02_document_analysis.R", package = "edgemodelr"))
analyzer <- DocumentAnalyzer()
results <- analyzer$analyze_documents(your_texts)
```

## 📚 Choose Your Path

- **Need it working NOW** → Start with `working_document_analysis.R`
- **Building production app** → Use `02_document_analysis.R`
- **Learning about models** → Check `01_model_comparison.R`
- **Having download issues** → Use `manual_setup_example.R`

## 🆚 Quick vs Professional Examples

| Aspect | Quick Examples | Professional Examples |
|--------|----------------|----------------------|
| **Goal** | Solve immediate problem | Production-ready systems |
| **Complexity** | Simple, direct | Comprehensive, feature-rich |
| **Error Handling** | Basic | Extensive |
| **Documentation** | Minimal | Complete |

## 📖 Additional Documentation

- `../../MODERN_MODELS.md` - Guide to 2024 small language models
- `../../EXAMPLES_GUIDE.md` - Complete navigation guide

---

All examples include error handling and work with the latest 2024 quantized models including Llama 3.2, Phi-3.5, Qwen2.5, and Gemma 2.