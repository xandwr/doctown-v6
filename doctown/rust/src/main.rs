// the entry point for ingest + later processing

use clap::Parser;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::io;
use std::path::Path;
use tempfile::TempDir;
use zip::ZipArchive;

#[derive(Parser, Debug)]
#[command(name = "doctown")]
#[command(about = "Ingest a GitHub repo and output its file structure as JSON")]
struct Args {
    /// GitHub repo URL (e.g., https://github.com/owner/repo)
    /// or path to a local zip file
    input: String,

    /// Branch to download (default: main)
    #[arg(short, long, default_value = "main")]
    branch: String,

    /// Pretty print JSON output
    #[arg(short, long, default_value = "false")]
    pretty: bool,

    /// Enable chunking and output chunks.json
    #[arg(short, long, default_value = "false")]
    chunks: bool,

    /// Chunk size in characters (default: 240)
    #[arg(long, default_value = "240")]
    chunk_size: usize,

    /// Output directory for JSON files (writes filestructure.json and chunks.json)
    /// If not specified, outputs to stdout
    #[arg(short, long)]
    output_dir: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "type")]
enum FileNode {
    #[serde(rename = "file")]
    File { name: String, size: u64 },
    #[serde(rename = "directory")]
    Directory {
        name: String,
        children: Vec<FileNode>,
    },
}

#[derive(Serialize, Deserialize, Debug)]
struct RepoStructure {
    repo_url: String,
    branch: String,
    structure: FileNode,
}

#[derive(Serialize, Deserialize, Debug)]
struct Chunk {
    chunk_id: String,
    file_path: String,
    start: usize,
    end: usize,
    text: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct ChunksOutput {
    repo_url: String,
    branch: String,
    total_chunks: usize,
    chunks: Vec<Chunk>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Create a temporary directory that will be automatically cleaned up
    let temp_dir = TempDir::new()?;
    let temp_path = temp_dir.path();

    // Download or copy the zip file
    let zip_path = temp_path.join("repo.zip");

    if args.input.starts_with("http://") || args.input.starts_with("https://") {
        // Convert GitHub URL to zip download URL
        let zip_url = github_url_to_zip(&args.input, &args.branch)?;
        download_file(&zip_url, &zip_path)?;
    } else {
        // Assume it's a local file path
        fs::copy(&args.input, &zip_path)?;
    }

    // Extract the zip file
    let extract_path = temp_path.join("extracted");
    fs::create_dir_all(&extract_path)?;
    extract_zip(&zip_path, &extract_path)?;

    // Find the root directory (GitHub zips have a single root folder)
    let root_dir = find_root_directory(&extract_path)?;

    // Build the file structure
    let structure =
        build_file_structure(&root_dir, root_dir.file_name().unwrap().to_str().unwrap())?;

    let repo_structure = RepoStructure {
        repo_url: args.input.clone(),
        branch: args.branch.clone(),
        structure,
    };

    // Output filestructure JSON
    let filestructure_json = if args.pretty {
        serde_json::to_string_pretty(&repo_structure)?
    } else {
        serde_json::to_string(&repo_structure)?
    };

    // Generate chunks if enabled
    let chunks_json = if args.chunks {
        eprintln!("Generating chunks...");
        let chunks = extract_chunks(&root_dir, args.chunk_size)?;
        let chunks_output = ChunksOutput {
            repo_url: args.input.clone(),
            branch: args.branch.clone(),
            total_chunks: chunks.len(),
            chunks,
        };

        Some(if args.pretty {
            serde_json::to_string_pretty(&chunks_output)?
        } else {
            serde_json::to_string(&chunks_output)?
        })
    } else {
        None
    };

    // Output to files or stdout
    if let Some(output_dir) = &args.output_dir {
        let output_path = Path::new(output_dir);
        fs::create_dir_all(output_path)?;

        // Write filestructure.json
        let filestructure_path = output_path.join("filestructure.json");
        fs::write(&filestructure_path, &filestructure_json)?;
        eprintln!("Wrote: {}", filestructure_path.display());

        // Write chunks.json if generated
        if let Some(chunks) = chunks_json {
            let chunks_path = output_path.join("chunks.json");
            fs::write(&chunks_path, &chunks)?;
            eprintln!("Wrote: {}", chunks_path.display());
        }
    } else {
        // Legacy stdout mode
        println!("{}", filestructure_json);
        if let Some(chunks) = chunks_json {
            eprintln!("\n--- CHUNKS OUTPUT ---");
            println!("{}", chunks);
        }
    }

    // temp_dir is automatically cleaned up when it goes out of scope
    Ok(())
}

/// Convert a GitHub repository URL to a zip download URL
fn github_url_to_zip(url: &str, branch: &str) -> Result<String, Box<dyn std::error::Error>> {
    // Handle various GitHub URL formats
    let url = url.trim_end_matches('/');
    let url = url.trim_end_matches(".git");

    // Extract owner/repo from URL
    let parts: Vec<&str> = url.split('/').collect();
    if parts.len() < 2 {
        return Err("Invalid GitHub URL format".into());
    }

    let owner = parts[parts.len() - 2];
    let repo = parts[parts.len() - 1];

    Ok(format!(
        "https://github.com/{}/{}/archive/refs/heads/{}.zip",
        owner, repo, branch
    ))
}

/// Download a file from a URL
fn download_file(url: &str, dest: &Path) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("Downloading from: {}", url);

    let response = reqwest::blocking::get(url)?;

    if !response.status().is_success() {
        return Err(format!("Failed to download: HTTP {}", response.status()).into());
    }

    let bytes = response.bytes()?;
    fs::write(dest, &bytes)?;

    eprintln!("Downloaded {} bytes", bytes.len());
    Ok(())
}

/// Extract a zip file to a destination directory
fn extract_zip(zip_path: &Path, dest: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let file = fs::File::open(zip_path)?;
    let mut archive = ZipArchive::new(file)?;

    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        let outpath = match file.enclosed_name() {
            Some(path) => dest.join(path),
            None => continue,
        };

        if file.name().ends_with('/') {
            fs::create_dir_all(&outpath)?;
        } else {
            if let Some(p) = outpath.parent() {
                if !p.exists() {
                    fs::create_dir_all(p)?;
                }
            }
            let mut outfile = fs::File::create(&outpath)?;
            io::copy(&mut file, &mut outfile)?;
        }
    }

    eprintln!("Extracted {} files", archive.len());
    Ok(())
}

/// Find the root directory in the extracted zip (GitHub creates a single root folder)
fn find_root_directory(
    extract_path: &Path,
) -> Result<std::path::PathBuf, Box<dyn std::error::Error>> {
    let entries: Vec<_> = fs::read_dir(extract_path)?.filter_map(|e| e.ok()).collect();

    if entries.len() == 1 && entries[0].path().is_dir() {
        Ok(entries[0].path())
    } else {
        // If there's no single root folder, use the extract path itself
        Ok(extract_path.to_path_buf())
    }
}

/// Recursively build the file structure tree
fn build_file_structure(path: &Path, name: &str) -> Result<FileNode, Box<dyn std::error::Error>> {
    if path.is_file() {
        let metadata = fs::metadata(path)?;
        Ok(FileNode::File {
            name: name.to_string(),
            size: metadata.len(),
        })
    } else {
        let mut children = Vec::new();

        let mut entries: Vec<_> = fs::read_dir(path)?.filter_map(|e| e.ok()).collect();

        // Sort entries: directories first, then files, alphabetically
        entries.sort_by(|a, b| {
            let a_is_dir = a.path().is_dir();
            let b_is_dir = b.path().is_dir();
            match (a_is_dir, b_is_dir) {
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                _ => a.file_name().cmp(&b.file_name()),
            }
        });

        for entry in entries {
            let entry_path = entry.path();
            let entry_name = entry.file_name().to_string_lossy().to_string();

            // Skip hidden files and common non-essential directories
            if entry_name.starts_with('.') {
                continue;
            }

            let child = build_file_structure(&entry_path, &entry_name)?;
            children.push(child);
        }

        Ok(FileNode::Directory {
            name: name.to_string(),
            children,
        })
    }
}

/// Extract and chunk text content from all files in the repository
fn extract_chunks(root_dir: &Path, chunk_size: usize) -> Result<Vec<Chunk>, Box<dyn std::error::Error>> {
    let mut all_chunks = Vec::new();
    let mut chunk_counter = 0;

    // Walk through all files recursively
    visit_files(root_dir, root_dir, &mut |file_path, relative_path| {
        // Try to read as UTF-8 text
        if let Ok(content) = fs::read_to_string(file_path) {
            // Skip if file is too small or empty
            if content.trim().is_empty() {
                return;
            }

            // Create a stable hash of the file path for consistent chunk IDs
            let mut hasher = Sha256::new();
            hasher.update(relative_path.as_bytes());
            let path_hash = format!("{:x}", hasher.finalize());
            let short_hash = &path_hash[..8];

            // Split content into chunks
            let chars: Vec<char> = content.chars().collect();
            let mut start = 0;

            while start < chars.len() {
                let end = std::cmp::min(start + chunk_size, chars.len());
                let chunk_text: String = chars[start..end].iter().collect();

                // Generate stable chunk ID
                let chunk_id = format!("chunk_{}_{:06}", short_hash, chunk_counter);
                chunk_counter += 1;

                all_chunks.push(Chunk {
                    chunk_id,
                    file_path: relative_path.clone(),
                    start,
                    end,
                    text: chunk_text,
                });

                start = end;
            }
        }
    })?;

    eprintln!("Extracted {} chunks from repository", all_chunks.len());
    Ok(all_chunks)
}

/// Recursively visit all files in a directory
fn visit_files<F>(
    path: &Path,
    root: &Path,
    callback: &mut F,
) -> Result<(), Box<dyn std::error::Error>>
where
    F: FnMut(&Path, String),
{
    if path.is_file() {
        let relative = path.strip_prefix(root)
            .unwrap_or(path)
            .to_string_lossy()
            .to_string();
        callback(path, relative);
    } else if path.is_dir() {
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let entry_path = entry.path();
            let entry_name = entry.file_name().to_string_lossy().to_string();

            // Skip hidden files and directories
            if entry_name.starts_with('.') {
                continue;
            }

            // Skip common binary/build directories
            if entry_path.is_dir() && matches!(entry_name.as_str(), "target" | "node_modules" | "dist" | "build" | "__pycache__") {
                continue;
            }

            visit_files(&entry_path, root, callback)?;
        }
    }

    Ok(())
}
