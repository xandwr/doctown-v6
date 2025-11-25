# Static Analysis Enrichment Roadmap

**Goal**: Build a semantic static analysis suite where LLMs receive pre-analyzed, structured code metadata instead of having to infer everything from raw text.

**Date**: November 25, 2025  
**Status**: Design Phase

---

## 🎯 Core Philosophy

> **"Extract, Don't Infer"**
> 
> The LLM should document what code does, not figure out what it is. Static analysis tools should provide:
> - Function signatures, parameters, return types
> - Call graphs and usage patterns
> - Type hierarchies and inheritance
> - Import/dependency relationships
> - Visibility and scope information

---

## 📊 Current State Analysis

### ✅ What Works
- **Universal chunk schema**: `UniversalChunk` is domain-agnostic and extensible
- **Metadata infrastructure**: `ChunkMetadata` supports rich contextual information
- **Rust ingestion**: Basic text chunking from GitHub repos
- **Python pipeline**: Converts chunks → embeddings → LLM docs

### ❌ Current Limitations

#### 1. **Dumb Text Chunking**
```rust
// Current: Just split by character count (240 chars)
let chunk_text: String = chars[start..end].iter().collect();
```
**Problem**: No semantic awareness. A chunk might be:
- Half a function
- Multiple unrelated snippets
- Missing critical context (function signature without body)

#### 2. **No Symbol Extraction**
```python
# Current metadata is mostly empty
metadata=ChunkMetadata(
    language="rust",
    line_start=45,
    line_end=67,
    # Missing: function_name, parameters, return_type, 
    #          parent_class, visibility, decorators, etc.
)
```

#### 3. **Non-Human-Readable IDs**
```json
{
  "symbol_id": "chunk_6a857f48_000060",  // ❌ Useless for humans
  "summary": "Represents a list of files..."
}
```

#### 4. **LLM Has to Guess Everything**
```python
# LLM prompt currently:
"""
Symbol: <raw text blob>
Type: text
Location: filestructure.json:1-1

{240 characters of JSON...}
"""
# LLM must infer: What is this? A function? A class? What does it do?
```

---

## 🚀 Implementation Roadmap

### **Phase 1: Symbol-Aware Chunking** (High Priority)
**Goal**: Extract actual code symbols (functions, classes, methods) instead of arbitrary character chunks.

#### 1.1 Use Tree-sitter for Multi-Language Parsing
**Why Tree-sitter?**
- Fast, incremental parser
- Supports 40+ languages out of the box
- Rust library available: `tree-sitter`
- Already used by GitHub, Atom, Neovim

**Languages to support**:
- ✅ Rust (`tree-sitter-rust`)
- ✅ Python (`tree-sitter-python`)
- ✅ TypeScript/JavaScript (`tree-sitter-typescript`)
- ✅ Go (`tree-sitter-go`)
- ✅ Java (`tree-sitter-java`)
- ✅ C/C++ (`tree-sitter-cpp`)

**Example Output**:
```json
{
  "chunk_id": "chunk_abc123",
  "symbol_name": "Session::new",
  "symbol_kind": "method",
  "path": "src/session.rs",
  "text": "pub fn new(env: &Environment) -> Result<Self> { ... }",
  "metadata": {
    "language": "rust",
    "line_start": 45,
    "line_end": 67,
    "visibility": "public",
    "parent_name": "Session",
    "parent_type": "impl",
    "parameters": [
      {"name": "env", "type": "&Environment"}
    ],
    "return_type": "Result<Self>",
    "is_async": false,
    "is_unsafe": false
  }
}
```

#### 1.2 Enhanced Chunk Schema
Extend `ChunkMetadata` to include:

```python
@dataclass
class ChunkMetadata:
    # ... existing fields ...
    
    # NEW: Symbol identification
    symbol_name: Optional[str] = None  # e.g., "calculate_total"
    qualified_name: Optional[str] = None  # e.g., "Calculator::calculate_total"
    symbol_kind: Optional[str] = None  # function, method, class, struct, etc.
    
    # NEW: Signature information (code domain)
    parameters: Optional[list[dict]] = None  # [{name, type, default}, ...]
    return_type: Optional[str] = None
    type_parameters: Optional[list[str]] = None  # Generic/template params
    
    # NEW: Modifiers and attributes
    modifiers: Optional[list[str]] = None  # async, unsafe, static, const
    decorators: Optional[list[str]] = None  # @property, #[derive(Debug)]
    annotations: Optional[list[str]] = None  # Type hints, attributes
    
    # NEW: Relationships
    implements: Optional[list[str]] = None  # Traits, interfaces
    extends: Optional[str] = None  # Base class/parent
    called_by: Optional[list[str]] = None  # Callers (from call graph)
    calls: Optional[list[str]] = None  # Callees
    uses_types: Optional[list[str]] = None  # Referenced types
```

#### 1.3 Rust Implementation Plan

**File**: `doctown/rust/src/parser.rs` (NEW)

```rust
use tree_sitter::{Parser, Language, Node, Tree};

pub struct SymbolExtractor {
    parser: Parser,
    language: Language,
}

pub struct Symbol {
    pub name: String,
    pub qualified_name: String,
    pub kind: SymbolKind,
    pub visibility: Visibility,
    pub line_start: usize,
    pub line_end: usize,
    pub text: String,
    pub signature: Option<Signature>,
    pub parent: Option<String>,
}

pub enum SymbolKind {
    Function,
    Method,
    Class,
    Struct,
    Enum,
    Trait,
    Module,
    Const,
    Variable,
}

pub struct Signature {
    pub parameters: Vec<Parameter>,
    pub return_type: Option<String>,
    pub type_params: Vec<String>,
    pub modifiers: Vec<String>,
}

impl SymbolExtractor {
    pub fn new(language: Language) -> Self { ... }
    
    pub fn extract_symbols(&mut self, source_code: &str) -> Vec<Symbol> {
        let tree = self.parser.parse(source_code, None).unwrap();
        let root_node = tree.root_node();
        
        let mut symbols = Vec::new();
        self.visit_node(root_node, source_code, &mut symbols, None);
        symbols
    }
    
    fn visit_node(&self, node: Node, source: &str, symbols: &mut Vec<Symbol>, parent: Option<&str>) {
        match node.kind() {
            "function_item" => self.extract_function(node, source, symbols, parent),
            "impl_item" => self.extract_impl(node, source, symbols),
            "struct_item" => self.extract_struct(node, source, symbols),
            // ... handle all symbol types
            _ => {
                // Recursively visit children
                for child in node.children(&mut node.walk()) {
                    self.visit_node(child, source, symbols, parent);
                }
            }
        }
    }
    
    fn extract_function(&self, node: Node, source: &str, symbols: &mut Vec<Symbol>, parent: Option<&str>) {
        // Extract function name, parameters, return type, etc.
        let name = self.get_child_by_field(node, "name")
            .map(|n| self.node_text(n, source))
            .unwrap_or_default();
        
        let visibility = self.get_visibility(node, source);
        let params = self.extract_parameters(node, source);
        let return_type = self.extract_return_type(node, source);
        
        symbols.push(Symbol {
            name: name.clone(),
            qualified_name: self.build_qualified_name(parent, &name),
            kind: SymbolKind::Function,
            visibility,
            line_start: node.start_position().row + 1,
            line_end: node.end_position().row + 1,
            text: self.node_text(node, source),
            signature: Some(Signature {
                parameters: params,
                return_type,
                type_params: vec![],
                modifiers: vec![],
            }),
            parent: parent.map(String::from),
        });
    }
}
```

**Benefits**:
- ✅ Each chunk is a complete, meaningful symbol
- ✅ Rich metadata for LLM context
- ✅ Human-readable symbol names
- ✅ Accurate line ranges
- ✅ Relationship information

---

### **Phase 2: Call Graph Analysis** (Medium Priority)
**Goal**: Map who calls whom, what uses what.

#### 2.1 Static Call Graph Construction
**Tools**:
- Rust: `rust-analyzer` LSP, `cargo-tree` for deps
- Python: `ast` module, `pycg` call graph generator
- TypeScript: TypeScript compiler API

**Output Example**:
```json
{
  "symbol_name": "Session::run",
  "calls": [
    "Allocator::allocate",
    "Tensor::from_array",
    "validate_input"
  ],
  "called_by": [
    "main::run_inference",
    "tests::test_session_run"
  ],
  "call_count_estimate": 12  // Static count of call sites
}
```

#### 2.2 Usage Pattern Detection
Identify common patterns:
- **Constructor patterns**: Functions that always call `new()`
- **Error handlers**: Functions in error propagation chains
- **Public API surface**: Functions called from outside the module
- **Internal helpers**: Functions only called internally

**Use Case**: LLM knows "This is a public API method" vs "This is an internal helper"

---

### **Phase 3: Type System Integration** (Medium Priority)
**Goal**: Provide complete type information to the LLM.

#### 3.1 Type Inference and Propagation
**For Rust**:
- Parse type annotations directly from code
- Use `rust-analyzer` for full type inference
- Extract trait bounds and lifetimes

**For Python**:
- Parse type hints (PEP 484)
- Infer types from usage (mypy, pyright)
- Handle dynamic typing gracefully

**For TypeScript**:
- Use TypeScript compiler for full type info
- Extract interface definitions
- Map type aliases and unions

**Output Example**:
```json
{
  "symbol_name": "calculate_total",
  "parameters": [
    {
      "name": "items",
      "type": "&[Item]",
      "type_kind": "slice_reference",
      "inner_type": "Item",
      "is_borrowed": true,
      "lifetime": "'a"
    }
  ],
  "return_type": "f64",
  "constraints": ["Item: Priceable"],
  "inferred_types": {
    "local_sum": "f64",
    "item_price": "f64"
  }
}
```

#### 3.2 Type Hierarchy Extraction
Map inheritance/trait relationships:

```json
{
  "type_name": "HttpClient",
  "implements": ["Client", "Send", "Sync"],
  "type_hierarchy": [
    "std::marker::Send",
    "std::marker::Sync",
    "crate::traits::Client"
  ],
  "associated_types": {
    "Request": "HttpRequest",
    "Response": "HttpResponse"
  }
}
```

---

### **Phase 4: Documentation Extraction** (Low Priority)
**Goal**: Don't make the LLM generate docs that already exist!

#### 4.1 Extract Existing Documentation
**Parse**:
- Rust: `///` doc comments, `#[doc = "..."]`
- Python: Docstrings (Google, NumPy, Sphinx styles)
- JavaScript: JSDoc comments
- Java: JavaDoc

**Use Case**: If a function has a docstring, include it in the chunk metadata. The LLM can:
- Validate existing docs
- Expand on them
- Point out inconsistencies

**Example**:
```json
{
  "symbol_name": "calculate_total",
  "existing_docs": {
    "summary": "Calculate the total price of items",
    "params": ["items: List of items to sum"],
    "returns": "Total price as a float",
    "examples": [">>> calculate_total([Item(10), Item(20)])\\n30.0"]
  },
  "llm_task": "validate_and_expand"  // vs "generate_new"
}
```

---

### **Phase 5: Cross-Reference Analysis** (Low Priority)
**Goal**: Map relationships between modules, files, and symbols.

#### 5.1 Import/Dependency Graph
```json
{
  "file": "src/session.rs",
  "imports": [
    {"module": "std::sync::Arc", "symbols": ["Arc"]},
    {"module": "crate::tensor", "symbols": ["Tensor", "TensorAllocator"]},
    {"module": "crate::environment", "symbols": ["Environment"]}
  ],
  "imported_by": [
    "src/inference.rs",
    "src/training.rs"
  ],
  "dependency_depth": 3  // How deep in dependency tree
}
```

#### 5.2 Symbol Reference Tracking
For each symbol, track:
- **Definition location**: Where it's defined
- **Usage locations**: All places it's used
- **Export status**: Is it part of public API?
- **Re-exports**: Where else is it exposed?

---

## 🎨 Enhanced LLM Prompt Template

**Before** (Current):
```
Symbol: <raw text>
Type: text
Location: filestructure.json:1-1

{240 chars of JSON...}
```

**After** (With Static Analysis):
```
=== CODE SYMBOL ===
Name: Session::run
Qualified: ort::session::Session::run
Kind: method (public)
Location: src/session.rs:L45-67

=== SIGNATURE ===
pub async fn run<T: Into<Tensor>>(
    &mut self,
    inputs: Vec<T>,
    options: RunOptions
) -> Result<Vec<Tensor>, OrtError>

Parameters:
  - self: &mut Self (mutable reference)
  - inputs: Vec<T> where T: Into<Tensor>
  - options: RunOptions (default: RunOptions::default())
Returns: Result<Vec<Tensor>, OrtError>
Modifiers: async, pub

=== CONTEXT ===
Parent: impl Session
Implements: Runner trait
Called by: 
  - inference::run_model (12 call sites)
  - training::evaluate (3 call sites)
Calls:
  - Allocator::allocate (for output tensors)
  - validate_inputs (input validation)
  - backend::execute (actual execution)

=== EXISTING DOCS ===
Summary: "Runs inference on the model with provided inputs"
[Some existing documentation found]

=== SOURCE CODE ===
{full function body}

=== YOUR TASK ===
Generate comprehensive documentation for this method.
Focus on:
- What the method does (high-level behavior)
- When to use it vs. alternatives
- Error conditions and handling
- Performance considerations
DO NOT re-describe the signature (we already extracted it).
```

---

## 📦 Implementation Strategy

### Phase 1: Quick Wins (Week 1-2)
1. ✅ Add `symbol_name` and `qualified_name` to `ChunkMetadata`
2. ✅ Integrate Tree-sitter for Rust parsing
3. ✅ Extract function/method signatures
4. ✅ Update LLM prompt to include symbol metadata
5. ✅ Expose symbol names in `documentation.json`

### Phase 2: Core Analysis (Week 3-4)
1. ⚙️ Add call graph extraction (basic)
2. ⚙️ Extract type information from signatures
3. ⚙️ Build parent-child relationships (class → methods)
4. ⚙️ Support Python and TypeScript parsing

### Phase 3: Advanced Features (Week 5-6)
1. 🔧 Full type inference integration
2. 🔧 Cross-reference mapping
3. 🔧 Import/dependency graphs
4. 🔧 Extract existing documentation

### Phase 4: Polish & Optimization (Week 7-8)
1. 🎨 Performance optimization
2. 🎨 Error handling and fallbacks
3. 🎨 Support for more languages
4. 🎨 Comprehensive test suite

---

## 🧪 Success Metrics

### Quantitative
- **Symbol extraction accuracy**: >95% of functions/classes detected
- **Chunk relevance**: Each chunk is a complete symbol (not split mid-function)
- **Metadata completeness**: >80% of chunks have full signature info
- **LLM cost reduction**: 30-40% fewer tokens (less inferring, more documenting)

### Qualitative
- **Human readability**: `symbol_id` reveals what the symbol is
- **LLM output quality**: More accurate, detailed documentation
- **Developer experience**: Easier to navigate and understand docs

---

## 🔧 Technical Decisions

### Why Tree-sitter over LSP?
- **Tree-sitter**: Fast, language-agnostic, designed for parsing
- **LSP**: Requires full language server setup, slower, more complex
- **Decision**: Use Tree-sitter for extraction, optionally integrate LSP for advanced type inference

### Why Not Use Language-Specific Tools?
We will! But as a second layer:
- **Layer 1**: Tree-sitter for universal symbol extraction
- **Layer 2**: Language-specific tools (rust-analyzer, pyright) for deep analysis

### Handling Dynamic Languages (Python, JavaScript)
- Extract what we can statically (function definitions, type hints)
- Mark dynamic aspects in metadata
- Let LLM handle ambiguity with context

---

## 🚦 Next Steps

### Immediate Actions
1. **Create `doctown/rust/src/parser.rs`**: Symbol extraction with Tree-sitter
2. **Update `ChunkMetadata`**: Add new fields for symbol info
3. **Modify `main.rs`**: Switch from character chunking to symbol extraction
4. **Update Python pipeline**: Handle new metadata fields
5. **Enhance LLM prompt**: Include symbol metadata in prompt

### Testing
1. Run on `ort` repository (current test case)
2. Compare before/after documentation quality
3. Measure token usage and cost
4. Validate symbol extraction accuracy

---

## 📚 References

- [Tree-sitter Documentation](https://tree-sitter.github.io/tree-sitter/)
- [Tree-sitter Rust](https://github.com/tree-sitter/tree-sitter-rust)
- [Rust Analyzer](https://rust-analyzer.github.io/)
- [Python AST Module](https://docs.python.org/3/library/ast.html)
- [TypeScript Compiler API](https://github.com/microsoft/TypeScript/wiki/Using-the-Compiler-API)

---

**End of Roadmap**  
*Last Updated: November 25, 2025*
