// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Remote filesystem storage for G4 tier (POSIX-based).
//!
//! This module provides [`RemoteFsStorage`] - a file-backed storage type for offloading
//! KV cache blocks to a network-mounted filesystem (3FS FUSE, NFS, Lustre, etc.).
//!
//! Unlike [`DiskStorage`] (G3), this storage:
//! - Uses a separate configurable directory (not the local disk cache dir)
//! - Always disables `O_DIRECT` (network mounts typically don't support it)
//! - Always uses zero-fill fallback (no `fallocate` since network FS may not support it)
//! - Is marked as [`Local`] (file-backed via local fd, like [`DiskStorage`])

use super::*;

use core::ffi::c_char;
use nix::unistd::{ftruncate, unlink};
use std::ffi::CStr;
use std::ffi::CString;
use std::fs::File;
use std::io::Write;
use std::os::unix::io::{FromRawFd, RawFd};
use std::path::{Path, PathBuf};
use std::sync::Arc;

const REMOTE_FS_CACHE_DIR_KEY: &str = "DYN_KVBM_REMOTE_FS_CACHE_DIR";
const DEFAULT_REMOTE_FS_CACHE_DIR: &str = "/mnt/3fs/kvbm_cache/";

const ZERO_BUF_SIZE: usize = 16 * 1024 * 1024; // 16MB

#[derive(Debug)]
pub struct RemoteFsStorage {
    fd: u64,
    file_name: String,
    size: usize,
    handles: RegistrationHandles,
    unlinked: bool,
}

impl Local for RemoteFsStorage {}

fn zero_fill_file(fd: RawFd, size: u64) -> anyhow::Result<()> {
    let buf = vec![0u8; ZERO_BUF_SIZE];
    let mut file = unsafe { File::from_raw_fd(nix::unistd::dup(fd).map_err(anyhow::Error::from)?) };

    let mut written: u64 = 0;
    while written < size {
        let remaining = size - written;
        let to_write = std::cmp::min(remaining as usize, buf.len());

        match file.write(&buf[..to_write]) {
            Ok(n) => {
                if n != to_write {
                    anyhow::bail!(
                        "Partial write: expected {} bytes, wrote {} bytes (total {}/{})",
                        to_write,
                        n,
                        written + n as u64,
                        size
                    );
                }
                written += n as u64;
            }
            Err(e) => {
                anyhow::bail!("Zero-fill write failed: {}", e);
            }
        }
    }

    file.flush()
        .map_err(|e| anyhow::anyhow!("Failed to flush zero-filled file: {}", e))?;

    // Truncate to exact size if we over-wrote
    if written > size {
        ftruncate(fd, size as i64)
            .map_err(|e| anyhow::anyhow!("Failed to truncate to exact size: {}", e))?;
    }

    tracing::debug!(
        "Successfully zero-filled {} bytes for remote FS storage",
        size
    );
    Ok(())
}

impl RemoteFsStorage {
    pub fn new(size: usize) -> Result<Self, StorageError> {
        let specified_dir = std::env::var(REMOTE_FS_CACHE_DIR_KEY)
            .unwrap_or_else(|_| DEFAULT_REMOTE_FS_CACHE_DIR.to_string());
        Self::new_in_dir(&specified_dir, size)
    }

    pub fn new_in_dir(dir: &str, size: usize) -> Result<Self, StorageError> {
        let file_path = Path::new(dir).join("dynamo-kvbm-remote-fs-cache-XXXXXX");

        if !file_path.parent().unwrap().exists() {
            std::fs::create_dir_all(file_path.parent().unwrap()).map_err(|e| {
                StorageError::AllocationFailed(format!(
                    "Failed to create remote FS cache dir {}: {}",
                    dir, e
                ))
            })?;
        }

        tracing::debug!(
            "Allocating remote FS cache file at {}",
            file_path.display()
        );

        let template = CString::new(file_path.to_str().unwrap()).unwrap();
        let mut template_bytes = template.into_bytes_with_nul();

        // mkostemp creates a temp file opened O_RDWR (no O_DIRECT for network FS)
        let raw_fd = unsafe {
            nix::libc::mkostemp(
                template_bytes.as_mut_ptr() as *mut c_char,
                nix::libc::O_CLOEXEC,
            )
        };

        if raw_fd < 0 {
            let file_name = CStr::from_bytes_with_nul(template_bytes.as_slice())
                .unwrap()
                .to_str()
                .unwrap_or("<invalid utf8>");
            return Err(StorageError::AllocationFailed(format!(
                "Failed to create temp file {}: {}",
                file_name,
                std::io::Error::last_os_error()
            )));
        }

        // No O_DIRECT - network mounts typically don't support it

        let file_name = CStr::from_bytes_with_nul(template_bytes.as_slice())
            .unwrap()
            .to_str()
            .map_err(|e| {
                StorageError::AllocationFailed(format!("Failed to read temp file name: {}", e))
            })?
            .to_string();

        // Always zero-fill (no fallocate for network FS)
        zero_fill_file(raw_fd, size as u64).map_err(|e| {
            StorageError::AllocationFailed(format!(
                "Failed to zero-fill remote FS temp file: {}",
                e
            ))
        })?;

        tracing::info!(
            "RemoteFsStorage created: fd={}, file={}, size={} bytes",
            raw_fd,
            file_name,
            size
        );

        Ok(Self {
            fd: raw_fd as u64,
            file_name,
            size,
            handles: RegistrationHandles::new(),
            unlinked: false,
        })
    }

    pub fn fd(&self) -> u64 {
        self.fd
    }

    /// Unlink our temp file.
    /// After unlinking, the file will be automatically deleted when the fd is closed.
    pub fn unlink(&mut self) -> Result<(), StorageError> {
        if self.unlinked {
            return Ok(());
        }

        tracing::info!(
            "Unlinking remote FS temp file (fd={}, file={}). File will be deleted when fd closes.",
            self.fd,
            self.file_name
        );

        self.unlinked = true;

        unlink(self.file_name.as_str()).map_err(|e| {
            tracing::error!(
                "Failed to unlink remote FS temp file: fd={}, file={}, error={}",
                self.fd,
                self.file_name,
                e
            );
            StorageError::AllocationFailed(format!("Failed to unlink temp file: {}", e))
        })
    }

    pub fn unlinked(&self) -> bool {
        self.unlinked
    }
}

impl Drop for RemoteFsStorage {
    fn drop(&mut self) {
        tracing::warn!(
            "RemoteFsStorage being dropped: fd={}, file={}, size={} bytes, already_unlinked={}",
            self.fd,
            self.file_name,
            self.size,
            self.unlinked
        );

        self.handles.release();
        let _ = self.unlink();

        tracing::info!(
            "RemoteFsStorage dropped and cleaned up: fd={}, file={}",
            self.fd,
            self.file_name
        );
    }
}

impl Storage for RemoteFsStorage {
    fn storage_type(&self) -> StorageType {
        StorageType::RemoteFs(self.fd())
    }

    fn addr(&self) -> u64 {
        0
    }

    fn size(&self) -> usize {
        self.size
    }

    unsafe fn as_ptr(&self) -> *const u8 {
        std::ptr::null()
    }

    unsafe fn as_mut_ptr(&mut self) -> *mut u8 {
        std::ptr::null_mut()
    }
}

impl RegisterableStorage for RemoteFsStorage {
    fn register(
        &mut self,
        key: &str,
        handle: Box<dyn RegistationHandle>,
    ) -> Result<(), StorageError> {
        self.handles.register(key, handle)
    }

    fn is_registered(&self, key: &str) -> bool {
        self.handles.is_registered(key)
    }

    fn registration_handle(&self, key: &str) -> Option<&dyn RegistationHandle> {
        self.handles.registration_handle(key)
    }
}

/// Allocator for RemoteFsStorage with a configurable directory.
pub struct RemoteFsAllocator {
    dir: PathBuf,
}

impl RemoteFsAllocator {
    pub fn new(dir: PathBuf) -> Self {
        Self { dir }
    }
}

impl Default for RemoteFsAllocator {
    fn default() -> Self {
        let dir = std::env::var(REMOTE_FS_CACHE_DIR_KEY)
            .unwrap_or_else(|_| DEFAULT_REMOTE_FS_CACHE_DIR.to_string());
        Self {
            dir: PathBuf::from(dir),
        }
    }
}

impl StorageAllocator<RemoteFsStorage> for RemoteFsAllocator {
    fn allocate(&self, size: usize) -> Result<RemoteFsStorage, StorageError> {
        RemoteFsStorage::new_in_dir(self.dir.to_str().unwrap(), size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::io::RawFd;

    #[test]
    fn test_remote_fs_storage_creation() {
        // Use /tmp for testing
        let storage =
            RemoteFsStorage::new_in_dir("/tmp/kvbm-remote-fs-test/", 4096).unwrap();
        assert_eq!(storage.size(), 4096);
        assert!(!storage.unlinked());

        // Verify the file is actually the correct size
        let fd = storage.fd() as RawFd;
        let mut buf = vec![0u8; 4096];
        let bytes_read = nix::sys::uio::pread(fd, &mut buf, 0).unwrap();
        assert_eq!(bytes_read, 4096);
        assert!(buf.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_remote_fs_storage_type() {
        let storage =
            RemoteFsStorage::new_in_dir("/tmp/kvbm-remote-fs-test/", 1024).unwrap();
        match storage.storage_type() {
            StorageType::RemoteFs(fd) => assert_eq!(fd, storage.fd()),
            other => panic!("Expected StorageType::RemoteFs, got {:?}", other),
        }
    }

    #[test]
    fn test_remote_fs_allocator() {
        let allocator = RemoteFsAllocator::new(PathBuf::from("/tmp/kvbm-remote-fs-test/"));
        let storage = allocator.allocate(8192).unwrap();
        assert_eq!(storage.size(), 8192);
    }
}
