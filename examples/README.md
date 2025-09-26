# edgemodelr Examples

This directory contains comprehensive examples demonstrating all the key features and capabilities of the edgemodelr package. Each example file is self-contained and includes detailed explanations, best practices, and real-world use cases.

## 📁 Example Files

### [01_basic_usage.R](01_basic_usage.R)
**Fundamental operations for getting started**
- Model loading and setup
- Basic text generation
- Temperature and parameter control
- Context length management
- Error handling and validation
- Essential cleanup procedures

**Key Functions:** `edge_setup()`, `edge_load_model()`, `edge_completion()`, `edge_free_model()`

### [02_ollama_integration.R](02_ollama_integration.R)
**Complete Ollama integration workflow**
- Automatic model discovery
- Hash-based model loading
- Compatibility testing
- Advanced model management
- Cross-platform support
- Error handling for edge cases

**Key Functions:** `edge_find_ollama_models()`, `edge_load_ollama_model()`, `test_ollama_model_compatibility()`

### [03_streaming_generation.R](03_streaming_generation.R)
**Real-time text generation and interactive interfaces**
- Basic streaming with callbacks
- Interactive chat interfaces
- Controlled generation with stopping conditions
- Output collection and processing
- Performance comparisons
- Advanced callback techniques

**Key Functions:** `edge_stream_completion()`, `edge_chat_stream()`

### [04_performance_optimization.R](04_performance_optimization.R)
**Performance tuning and optimization strategies**
- Benchmarking and profiling
- Context length optimization
- Memory management best practices
- GPU acceleration testing
- Batch processing techniques
- Advanced performance monitoring

**Key Functions:** `edge_benchmark()`, `edge_clean_cache()`

## 🚀 Getting Started

### Prerequisites
1. **Install edgemodelr package**
   ```r
   # Install from source or your preferred method
   devtools::install()
   ```

2. **Download Models**
   - **GGUF Models**: Download from Hugging Face (e.g., Phi-3, Llama, CodeLlama)
   - **Ollama Models**: Install Ollama and download models
     ```bash
     # Install Ollama first: https://ollama.ai
     ollama pull llama3.2:latest
     ollama pull phi3:mini
     ```

### Quick Start
```r
library(edgemodelr)

# Method 1: Using GGUF files
setup <- edge_setup()
if (!is.null(setup)) {
  ctx <- edge_load_model(setup$available_models[1])
  result <- edge_completion(ctx, "Hello, world!", n_predict = 20)
  edge_free_model(ctx)
}

# Method 2: Using Ollama models
ollama_info <- edge_find_ollama_models()
if (!is.null(ollama_info)) {
  ctx <- edge_load_ollama_model("dde5aa3f")  # Use first 8 chars of SHA
  result <- edge_completion(ctx, "Hello, world!", n_predict = 20)
  edge_free_model(ctx)
}
```

## 📋 Example Overview

| Example | Focus Area | Difficulty | Runtime |
|---------|------------|------------|---------|
| 01_basic_usage | Core functionality | Beginner | 2-5 min |
| 02_ollama_integration | Ollama models | Intermediate | 3-7 min |
| 03_streaming_generation | Real-time generation | Intermediate | 5-10 min |
| 04_performance_optimization | Optimization | Advanced | 10-15 min |

## 🎯 Learning Path

### For Beginners
1. Start with **01_basic_usage.R** to understand core concepts
2. Learn model management with **02_ollama_integration.R**
3. Explore interactive features in **03_streaming_generation.R**

### For Advanced Users
1. Review **04_performance_optimization.R** for optimization techniques
2. Combine concepts from multiple examples for complex applications
3. Use examples as templates for your own projects

## 💡 Key Concepts Covered

### Model Management
- Loading and unloading models efficiently
- Memory management and cleanup
- Cross-platform compatibility
- Model discovery and validation

### Text Generation
- Prompt engineering techniques
- Parameter tuning (temperature, n_predict, etc.)
- Streaming vs batch generation
- Error handling and recovery

### Performance Optimization
- Benchmarking and profiling
- GPU acceleration setup
- Memory usage optimization
- Batch processing strategies

### Integration Patterns
- Ollama model integration
- Interactive applications
- Real-time processing
- Production deployment considerations

## 🛠️ Troubleshooting

### Common Issues

**No models found:**
```r
# Check model availability
setup <- edge_setup()
print(setup$available_models)

# Or check Ollama models
ollama_info <- edge_find_ollama_models()
print(ollama_info)
```

**Memory issues:**
```r
# Clear cache and free models
edge_clean_cache()
# Always call edge_free_model(ctx) when done
```

**Performance issues:**
```r
# Run benchmark to identify bottlenecks
ctx <- edge_load_model(model_path)
results <- edge_benchmark(ctx)
print(results)
edge_free_model(ctx)
```

**Ollama integration issues:**
- Ensure Ollama is installed and running
- Check model downloads: `ollama list`
- Verify model compatibility with `test_compatibility = TRUE`

### Getting Help
- Check function documentation: `?edge_load_model`
- Review example comments for detailed explanations
- Test with minimal examples before complex applications

## 🔧 Customization

### Adapting Examples
All examples are designed to be easily customizable:

1. **Change models**: Replace model paths/hashes with your models
2. **Adjust parameters**: Modify n_ctx, temperature, n_predict as needed
3. **Add functionality**: Extend examples with your specific use cases
4. **Error handling**: Add additional error checking for production use

### Creating New Examples
When creating new examples, follow the established pattern:
```r
# ==============================================================================
# Your Example Title for edgemodelr
# ==============================================================================
# Brief description of what this example demonstrates

library(edgemodelr)

# ------------------------------------------------------------------------------
# Example 1: Descriptive Name
# ------------------------------------------------------------------------------

cat("Example 1: Descriptive Name\n")
cat("============================\n\n")

# Your example code here
# Include error handling
# Add explanatory comments
# Show results and cleanup

cat("✅ Example 1 completed successfully!\n\n")
```

## 🚀 Next Steps

After working through these examples:

1. **Build Applications**: Use examples as building blocks for larger projects
2. **Optimize Performance**: Apply optimization techniques to your use cases
3. **Integrate Systems**: Combine edgemodelr with other R packages and tools
4. **Share Knowledge**: Contribute improvements back to the community

---

**Happy modeling with edgemodelr! 🤖✨**