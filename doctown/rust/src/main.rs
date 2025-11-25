// the entry point for ingest + later processing

use clap::Parser;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{self, Read};
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
        repo_url: args.input,
        branch: args.branch,
        structure,
    };

    // Output JSON
    let json = if args.pretty {
        serde_json::to_string_pretty(&repo_structure)?
    } else {
        serde_json::to_string(&repo_structure)?
    };

    println!("{}", json);

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
