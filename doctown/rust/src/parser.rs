/// Symbol extraction using Tree-sitter for multi-language parsing
///
/// This module provides semantic code analysis to extract actual symbols
/// (functions, classes, methods, structs) instead of arbitrary text chunks.

use serde::{Deserialize, Serialize};
use tree_sitter::{Language, Node, Parser};

/// Supported programming languages
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LanguageId {
    Rust,
    Python,
    TypeScript,
    JavaScript,
    Go,
    Java,
    C,
    Cpp,
}

impl LanguageId {
    /// Get the tree-sitter Language for this language
    pub fn get_language(&self) -> Language {
        match self {
            LanguageId::Rust => tree_sitter_rust::language(),
            LanguageId::Python => tree_sitter_python::language(),
            LanguageId::TypeScript => tree_sitter_typescript::language_typescript(),
            LanguageId::JavaScript => tree_sitter_typescript::language_tsx(),
            LanguageId::Go => tree_sitter_go::language(),
            LanguageId::Java => tree_sitter_java::language(),
            LanguageId::C => tree_sitter_c::language(),
            LanguageId::Cpp => tree_sitter_cpp::language(),
        }
    }

    /// Detect language from file extension
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "rs" => Some(LanguageId::Rust),
            "py" => Some(LanguageId::Python),
            "ts" => Some(LanguageId::TypeScript),
            "tsx" => Some(LanguageId::TypeScript),
            "js" => Some(LanguageId::JavaScript),
            "jsx" => Some(LanguageId::JavaScript),
            "go" => Some(LanguageId::Go),
            "java" => Some(LanguageId::Java),
            "c" | "h" => Some(LanguageId::C),
            "cpp" | "cc" | "cxx" | "hpp" | "hxx" => Some(LanguageId::Cpp),
            _ => None,
        }
    }

    #[allow(dead_code)]
    pub fn as_str(&self) -> &'static str {
        match self {
            LanguageId::Rust => "rust",
            LanguageId::Python => "python",
            LanguageId::TypeScript => "typescript",
            LanguageId::JavaScript => "javascript",
            LanguageId::Go => "go",
            LanguageId::Java => "java",
            LanguageId::C => "c",
            LanguageId::Cpp => "cpp",
        }
    }
}

/// Symbol visibility/access level
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Visibility {
    Public,
    Private,
    Protected,
    Internal,
    Package,
}

/// Type of code symbol
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum SymbolKind {
    Function,
    Method,
    Class,
    Struct,
    Enum,
    Trait,
    Interface,
    Module,
    Const,
    Variable,
    TypeAlias,
    Impl,
}

/// Parameter information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Parameter {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub type_annotation: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_value: Option<String>,
}

/// Function/method signature
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signature {
    pub parameters: Vec<Parameter>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub return_type: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub type_parameters: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub modifiers: Vec<String>,
}

/// A code symbol extracted from source
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Symbol {
    pub name: String,
    pub qualified_name: String,
    pub kind: SymbolKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub visibility: Option<Visibility>,
    pub line_start: usize,
    pub line_end: usize,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature: Option<Signature>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub doc_comment: Option<String>,
}

/// Symbol extractor for a specific language
pub struct SymbolExtractor {
    parser: Parser,
    language: LanguageId,
}

impl SymbolExtractor {
    /// Create a new symbol extractor for the given language
    pub fn new(language: LanguageId) -> Result<Self, String> {
        let mut parser = Parser::new();
        let ts_language = language.get_language();
        
        parser
            .set_language(&ts_language)
            .map_err(|e| format!("Failed to set language: {:?}", e))?;

        Ok(Self { parser, language })
    }

    /// Extract all symbols from source code
    pub fn extract_symbols(&mut self, source_code: &str) -> Result<Vec<Symbol>, String> {
        let tree = self
            .parser
            .parse(source_code, None)
            .ok_or_else(|| "Failed to parse source code".to_string())?;

        let root_node = tree.root_node();
        let mut symbols = Vec::new();

        self.visit_node(root_node, source_code, &mut symbols, None);

        Ok(symbols)
    }

    /// Recursively visit nodes to extract symbols
    fn visit_node(
        &self,
        node: Node,
        source: &str,
        symbols: &mut Vec<Symbol>,
        parent: Option<&str>,
    ) {
        match self.language {
            LanguageId::Rust => self.visit_rust_node(node, source, symbols, parent),
            LanguageId::Python => self.visit_python_node(node, source, symbols, parent),
            LanguageId::TypeScript | LanguageId::JavaScript => {
                self.visit_ts_node(node, source, symbols, parent)
            }
            _ => {
                // For other languages, recursively visit children
                let mut cursor = node.walk();
                for child in node.children(&mut cursor) {
                    self.visit_node(child, source, symbols, parent);
                }
            }
        }
    }

    /// Visit Rust-specific nodes
    fn visit_rust_node(
        &self,
        node: Node,
        source: &str,
        symbols: &mut Vec<Symbol>,
        parent: Option<&str>,
    ) {
        match node.kind() {
            "function_item" => {
                if let Some(symbol) = self.extract_rust_function(node, source, parent) {
                    symbols.push(symbol);
                }
            }
            "impl_item" => {
                // Extract impl block name and visit its methods
                let impl_name = self.extract_rust_impl_name(node, source);
                let mut cursor = node.walk();
                for child in node.children(&mut cursor) {
                    self.visit_node(child, source, symbols, impl_name.as_deref());
                }
            }
            "struct_item" => {
                if let Some(symbol) = self.extract_rust_struct(node, source, parent) {
                    symbols.push(symbol);
                }
            }
            "enum_item" => {
                if let Some(symbol) = self.extract_rust_enum(node, source, parent) {
                    symbols.push(symbol);
                }
            }
            "trait_item" => {
                if let Some(symbol) = self.extract_rust_trait(node, source, parent) {
                    symbols.push(symbol);
                }
            }
            "const_item" | "static_item" => {
                if let Some(symbol) = self.extract_rust_const(node, source, parent) {
                    symbols.push(symbol);
                }
            }
            _ => {
                // Recursively visit children
                let mut cursor = node.walk();
                for child in node.children(&mut cursor) {
                    self.visit_node(child, source, symbols, parent);
                }
            }
        }
    }

    /// Visit Python-specific nodes
    fn visit_python_node(
        &self,
        node: Node,
        source: &str,
        symbols: &mut Vec<Symbol>,
        parent: Option<&str>,
    ) {
        match node.kind() {
            "function_definition" => {
                if let Some(symbol) = self.extract_python_function(node, source, parent) {
                    symbols.push(symbol);
                }
            }
            "class_definition" => {
                if let Some(symbol) = self.extract_python_class(node, source, parent) {
                    // Extract class name and visit its methods
                    let class_name = symbol.name.clone();
                    symbols.push(symbol);
                    
                    // Visit class body for methods
                    if let Some(body) = node.child_by_field_name("body") {
                        let mut cursor = body.walk();
                        for child in body.children(&mut cursor) {
                            self.visit_node(child, source, symbols, Some(&class_name));
                        }
                    }
                }
            }
            _ => {
                // Recursively visit children
                let mut cursor = node.walk();
                for child in node.children(&mut cursor) {
                    self.visit_node(child, source, symbols, parent);
                }
            }
        }
    }

    /// Visit TypeScript/JavaScript-specific nodes
    fn visit_ts_node(
        &self,
        node: Node,
        source: &str,
        symbols: &mut Vec<Symbol>,
        parent: Option<&str>,
    ) {
        match node.kind() {
            "function_declaration" | "method_definition" => {
                if let Some(symbol) = self.extract_ts_function(node, source, parent) {
                    symbols.push(symbol);
                }
            }
            "class_declaration" => {
                if let Some(symbol) = self.extract_ts_class(node, source, parent) {
                    let class_name = symbol.name.clone();
                    symbols.push(symbol);
                    
                    // Visit class body
                    if let Some(body) = node.child_by_field_name("body") {
                        let mut cursor = body.walk();
                        for child in body.children(&mut cursor) {
                            self.visit_node(child, source, symbols, Some(&class_name));
                        }
                    }
                }
            }
            "interface_declaration" => {
                if let Some(symbol) = self.extract_ts_interface(node, source, parent) {
                    symbols.push(symbol);
                }
            }
            _ => {
                let mut cursor = node.walk();
                for child in node.children(&mut cursor) {
                    self.visit_node(child, source, symbols, parent);
                }
            }
        }
    }

    // ========== Rust Extractors ==========

    fn extract_rust_function(
        &self,
        node: Node,
        source: &str,
        parent: Option<&str>,
    ) -> Option<Symbol> {
        let name = node.child_by_field_name("name")?;
        let name_str = self.node_text(name, source);

        let visibility = self.extract_rust_visibility(node, source);
        let parameters = self.extract_rust_parameters(node, source);
        let return_type = self.extract_rust_return_type(node, source);
        let modifiers = self.extract_rust_modifiers(node, source);
        let doc_comment = self.extract_doc_comment_before(node, source);

        let kind = if parent.is_some() {
            SymbolKind::Method
        } else {
            SymbolKind::Function
        };

        let qualified_name = self.build_qualified_name(parent, &name_str);

        Some(Symbol {
            name: name_str,
            qualified_name,
            kind,
            visibility: Some(visibility),
            line_start: node.start_position().row + 1,
            line_end: node.end_position().row + 1,
            text: self.node_text(node, source),
            signature: Some(Signature {
                parameters,
                return_type,
                type_parameters: vec![],
                modifiers,
            }),
            parent: parent.map(String::from),
            doc_comment,
        })
    }

    fn extract_rust_impl_name(&self, node: Node, source: &str) -> Option<String> {
        // Look for the type being implemented
        node.child_by_field_name("type")
            .map(|n| self.node_text(n, source))
    }

    fn extract_rust_struct(&self, node: Node, source: &str, parent: Option<&str>) -> Option<Symbol> {
        let name = node.child_by_field_name("name")?;
        let name_str = self.node_text(name, source);
        let visibility = self.extract_rust_visibility(node, source);
        let doc_comment = self.extract_doc_comment_before(node, source);

        Some(Symbol {
            name: name_str.clone(),
            qualified_name: self.build_qualified_name(parent, &name_str),
            kind: SymbolKind::Struct,
            visibility: Some(visibility),
            line_start: node.start_position().row + 1,
            line_end: node.end_position().row + 1,
            text: self.node_text(node, source),
            signature: None,
            parent: parent.map(String::from),
            doc_comment,
        })
    }

    fn extract_rust_enum(&self, node: Node, source: &str, parent: Option<&str>) -> Option<Symbol> {
        let name = node.child_by_field_name("name")?;
        let name_str = self.node_text(name, source);
        let visibility = self.extract_rust_visibility(node, source);
        let doc_comment = self.extract_doc_comment_before(node, source);

        Some(Symbol {
            name: name_str.clone(),
            qualified_name: self.build_qualified_name(parent, &name_str),
            kind: SymbolKind::Enum,
            visibility: Some(visibility),
            line_start: node.start_position().row + 1,
            line_end: node.end_position().row + 1,
            text: self.node_text(node, source),
            signature: None,
            parent: parent.map(String::from),
            doc_comment,
        })
    }

    fn extract_rust_trait(&self, node: Node, source: &str, parent: Option<&str>) -> Option<Symbol> {
        let name = node.child_by_field_name("name")?;
        let name_str = self.node_text(name, source);
        let visibility = self.extract_rust_visibility(node, source);
        let doc_comment = self.extract_doc_comment_before(node, source);

        Some(Symbol {
            name: name_str.clone(),
            qualified_name: self.build_qualified_name(parent, &name_str),
            kind: SymbolKind::Trait,
            visibility: Some(visibility),
            line_start: node.start_position().row + 1,
            line_end: node.end_position().row + 1,
            text: self.node_text(node, source),
            signature: None,
            parent: parent.map(String::from),
            doc_comment,
        })
    }

    fn extract_rust_const(&self, node: Node, source: &str, parent: Option<&str>) -> Option<Symbol> {
        let name = node.child_by_field_name("name")?;
        let name_str = self.node_text(name, source);
        let visibility = self.extract_rust_visibility(node, source);

        Some(Symbol {
            name: name_str.clone(),
            qualified_name: self.build_qualified_name(parent, &name_str),
            kind: SymbolKind::Const,
            visibility: Some(visibility),
            line_start: node.start_position().row + 1,
            line_end: node.end_position().row + 1,
            text: self.node_text(node, source),
            signature: None,
            parent: parent.map(String::from),
            doc_comment: None,
        })
    }

    fn extract_rust_visibility(&self, node: Node, source: &str) -> Visibility {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "visibility_modifier" {
                let vis_text = self.node_text(child, source);
                return match vis_text.as_str() {
                    "pub" => Visibility::Public,
                    "pub(crate)" => Visibility::Internal,
                    _ => Visibility::Public,
                };
            }
        }
        Visibility::Private
    }

    fn extract_rust_parameters(&self, node: Node, source: &str) -> Vec<Parameter> {
        let mut params = Vec::new();
        
        if let Some(params_node) = node.child_by_field_name("parameters") {
            let mut cursor = params_node.walk();
            for child in params_node.children(&mut cursor) {
                if child.kind() == "parameter" {
                    if let Some(pattern) = child.child_by_field_name("pattern") {
                        let name = self.node_text(pattern, source);
                        let type_annotation = child
                            .child_by_field_name("type")
                            .map(|t| self.node_text(t, source));
                        
                        params.push(Parameter {
                            name,
                            type_annotation,
                            default_value: None,
                        });
                    }
                }
            }
        }
        
        params
    }

    fn extract_rust_return_type(&self, node: Node, source: &str) -> Option<String> {
        node.child_by_field_name("return_type")
            .map(|n| self.node_text(n, source).trim_start_matches("->").trim().to_string())
    }

    fn extract_rust_modifiers(&self, node: Node, _source: &str) -> Vec<String> {
        let mut modifiers = Vec::new();
        let mut cursor = node.walk();
        
        for child in node.children(&mut cursor) {
            match child.kind() {
                "async" => modifiers.push("async".to_string()),
                "unsafe" => modifiers.push("unsafe".to_string()),
                "const" => modifiers.push("const".to_string()),
                _ => {}
            }
        }
        
        modifiers
    }

    // ========== Python Extractors ==========

    fn extract_python_function(
        &self,
        node: Node,
        source: &str,
        parent: Option<&str>,
    ) -> Option<Symbol> {
        let name = node.child_by_field_name("name")?;
        let name_str = self.node_text(name, source);
        
        let parameters = self.extract_python_parameters(node, source);
        let return_type = self.extract_python_return_type(node, source);
        let modifiers = self.extract_python_decorators(node, source);
        let doc_comment = self.extract_python_docstring(node, source);
        
        let kind = if parent.is_some() {
            SymbolKind::Method
        } else {
            SymbolKind::Function
        };

        let visibility = if name_str.starts_with('_') {
            Visibility::Private
        } else {
            Visibility::Public
        };

        Some(Symbol {
            name: name_str.clone(),
            qualified_name: self.build_qualified_name(parent, &name_str),
            kind,
            visibility: Some(visibility),
            line_start: node.start_position().row + 1,
            line_end: node.end_position().row + 1,
            text: self.node_text(node, source),
            signature: Some(Signature {
                parameters,
                return_type,
                type_parameters: vec![],
                modifiers,
            }),
            parent: parent.map(String::from),
            doc_comment,
        })
    }

    fn extract_python_class(
        &self,
        node: Node,
        source: &str,
        parent: Option<&str>,
    ) -> Option<Symbol> {
        let name = node.child_by_field_name("name")?;
        let name_str = self.node_text(name, source);
        let doc_comment = self.extract_python_docstring(node, source);

        Some(Symbol {
            name: name_str.clone(),
            qualified_name: self.build_qualified_name(parent, &name_str),
            kind: SymbolKind::Class,
            visibility: Some(Visibility::Public),
            line_start: node.start_position().row + 1,
            line_end: node.end_position().row + 1,
            text: self.node_text(node, source),
            signature: None,
            parent: parent.map(String::from),
            doc_comment,
        })
    }

    fn extract_python_parameters(&self, node: Node, source: &str) -> Vec<Parameter> {
        let mut params = Vec::new();
        
        if let Some(params_node) = node.child_by_field_name("parameters") {
            let mut cursor = params_node.walk();
            for child in params_node.children(&mut cursor) {
                if child.kind() == "identifier" {
                    params.push(Parameter {
                        name: self.node_text(child, source),
                        type_annotation: None,
                        default_value: None,
                    });
                } else if child.kind() == "typed_parameter" {
                    let name = child.child(0).map(|n| self.node_text(n, source)).unwrap_or_default();
                    let type_annotation = child.child_by_field_name("type").map(|t| self.node_text(t, source));
                    params.push(Parameter {
                        name,
                        type_annotation,
                        default_value: None,
                    });
                } else if child.kind() == "default_parameter" {
                    let name = child.child(0).map(|n| self.node_text(n, source)).unwrap_or_default();
                    let default = child.child_by_field_name("value").map(|v| self.node_text(v, source));
                    params.push(Parameter {
                        name,
                        type_annotation: None,
                        default_value: default,
                    });
                }
            }
        }
        
        params
    }

    fn extract_python_return_type(&self, node: Node, source: &str) -> Option<String> {
        node.child_by_field_name("return_type")
            .map(|n| self.node_text(n, source).trim_start_matches("->").trim().to_string())
    }

    fn extract_python_decorators(&self, node: Node, source: &str) -> Vec<String> {
        let mut decorators = Vec::new();
        let mut cursor = node.walk();
        
        for child in node.children(&mut cursor) {
            if child.kind() == "decorator" {
                decorators.push(self.node_text(child, source));
            }
        }
        
        decorators
    }

    fn extract_python_docstring(&self, node: Node, source: &str) -> Option<String> {
        // Look for string in function body
        if let Some(body) = node.child_by_field_name("body") {
            if let Some(first_child) = body.child(0) {
                if first_child.kind() == "expression_statement" {
                    if let Some(string_node) = first_child.child(0) {
                        if string_node.kind() == "string" {
                            let text = self.node_text(string_node, source);
                            // Remove quotes
                            return Some(text.trim_matches(|c| c == '"' || c == '\'').to_string());
                        }
                    }
                }
            }
        }
        None
    }

    // ========== TypeScript/JavaScript Extractors ==========

    fn extract_ts_function(
        &self,
        node: Node,
        source: &str,
        parent: Option<&str>,
    ) -> Option<Symbol> {
        let name = node.child_by_field_name("name")?;
        let name_str = self.node_text(name, source);
        
        let parameters = self.extract_ts_parameters(node, source);
        let return_type = self.extract_ts_return_type(node, source);
        
        let kind = if node.kind() == "method_definition" {
            SymbolKind::Method
        } else {
            SymbolKind::Function
        };

        Some(Symbol {
            name: name_str.clone(),
            qualified_name: self.build_qualified_name(parent, &name_str),
            kind,
            visibility: Some(Visibility::Public),
            line_start: node.start_position().row + 1,
            line_end: node.end_position().row + 1,
            text: self.node_text(node, source),
            signature: Some(Signature {
                parameters,
                return_type,
                type_parameters: vec![],
                modifiers: vec![],
            }),
            parent: parent.map(String::from),
            doc_comment: None,
        })
    }

    fn extract_ts_class(
        &self,
        node: Node,
        source: &str,
        parent: Option<&str>,
    ) -> Option<Symbol> {
        let name = node.child_by_field_name("name")?;
        let name_str = self.node_text(name, source);

        Some(Symbol {
            name: name_str.clone(),
            qualified_name: self.build_qualified_name(parent, &name_str),
            kind: SymbolKind::Class,
            visibility: Some(Visibility::Public),
            line_start: node.start_position().row + 1,
            line_end: node.end_position().row + 1,
            text: self.node_text(node, source),
            signature: None,
            parent: parent.map(String::from),
            doc_comment: None,
        })
    }

    fn extract_ts_interface(
        &self,
        node: Node,
        source: &str,
        parent: Option<&str>,
    ) -> Option<Symbol> {
        let name = node.child_by_field_name("name")?;
        let name_str = self.node_text(name, source);

        Some(Symbol {
            name: name_str.clone(),
            qualified_name: self.build_qualified_name(parent, &name_str),
            kind: SymbolKind::Interface,
            visibility: Some(Visibility::Public),
            line_start: node.start_position().row + 1,
            line_end: node.end_position().row + 1,
            text: self.node_text(node, source),
            signature: None,
            parent: parent.map(String::from),
            doc_comment: None,
        })
    }

    fn extract_ts_parameters(&self, node: Node, source: &str) -> Vec<Parameter> {
        let mut params = Vec::new();
        
        if let Some(params_node) = node.child_by_field_name("parameters") {
            let mut cursor = params_node.walk();
            for child in params_node.children(&mut cursor) {
                if child.kind() == "required_parameter" || child.kind() == "optional_parameter" {
                    if let Some(pattern) = child.child_by_field_name("pattern") {
                        let name = self.node_text(pattern, source);
                        let type_annotation = child
                            .child_by_field_name("type")
                            .map(|t| self.node_text(t, source));
                        
                        params.push(Parameter {
                            name,
                            type_annotation,
                            default_value: None,
                        });
                    }
                }
            }
        }
        
        params
    }

    fn extract_ts_return_type(&self, node: Node, source: &str) -> Option<String> {
        node.child_by_field_name("return_type")
            .map(|n| self.node_text(n, source).trim_start_matches(':').trim().to_string())
    }

    // ========== Helper Methods ==========

    fn node_text(&self, node: Node, source: &str) -> String {
        node.utf8_text(source.as_bytes())
            .unwrap_or_default()
            .to_string()
    }

    fn build_qualified_name(&self, parent: Option<&str>, name: &str) -> String {
        match parent {
            Some(p) => format!("{}::{}", p, name),
            None => name.to_string(),
        }
    }

    fn extract_doc_comment_before(&self, node: Node, source: &str) -> Option<String> {
        // Look for comments immediately before the node
        let start_byte = node.start_byte();
        if start_byte < 4 {
            return None;
        }
        
        // Search backwards for doc comments (///, /** */)
        let before_text = &source[0..start_byte];
        let lines: Vec<&str> = before_text.lines().collect();
        
        let mut doc_lines = Vec::new();
        for line in lines.iter().rev() {
            let trimmed = line.trim();
            if trimmed.starts_with("///") {
                doc_lines.push(trimmed.trim_start_matches("///").trim());
            } else if !trimmed.is_empty() {
                break;
            }
        }
        
        if doc_lines.is_empty() {
            None
        } else {
            doc_lines.reverse();
            Some(doc_lines.join("\n"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rust_function_extraction() {
        let source = r#"
            /// Calculate the sum of items
            pub fn calculate_total(items: &[f64]) -> f64 {
                items.iter().sum()
            }
        "#;

        let mut extractor = SymbolExtractor::new(LanguageId::Rust).unwrap();
        let symbols = extractor.extract_symbols(source).unwrap();

        assert_eq!(symbols.len(), 1);
        assert_eq!(symbols[0].name, "calculate_total");
        assert_eq!(symbols[0].kind, SymbolKind::Function);
        assert_eq!(symbols[0].visibility, Some(Visibility::Public));
    }

    #[test]
    fn test_python_class_extraction() {
        let source = r#"
class Calculator:
    """A simple calculator"""
    
    def add(self, a: int, b: int) -> int:
        """Add two numbers"""
        return a + b
        "#;

        let mut extractor = SymbolExtractor::new(LanguageId::Python).unwrap();
        let symbols = extractor.extract_symbols(source).unwrap();

        assert!(symbols.len() >= 2); // Class + method
        assert!(symbols.iter().any(|s| s.name == "Calculator" && s.kind == SymbolKind::Class));
        assert!(symbols.iter().any(|s| s.name == "add" && s.kind == SymbolKind::Method));
    }
}
