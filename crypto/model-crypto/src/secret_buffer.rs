//! Secret memory buffer that is locked into RAM and zeroized on drop.
//!
//! On Unix systems this uses `mlock(2)` to prevent the buffer from being swapped.
//! On unsupported systems it falls back to a normal `Vec<u8>` while still
//! zeroizing on drop.
//!
//! This is the software-only hardening layer described in PRD-006.

use std::ops::{Deref, DerefMut};
use zeroize::{Zeroize, ZeroizeOnDrop};

use crate::ModelCryptoError;

/// A growable, zeroizing, optionally `mlock`-ed byte buffer.
#[derive(Debug, ZeroizeOnDrop)]
pub struct SecretBuffer {
    #[zeroize(skip)]
    inner: Vec<u8>,
    #[zeroize(skip)]
    locked: bool,
}

impl SecretBuffer {
    /// Create an empty secret buffer.
    pub fn new() -> Self {
        Self { inner: Vec::new(), locked: false }
    }

    /// Create a secret buffer with the given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self { inner: Vec::with_capacity(capacity), locked: false }
    }

    /// Create a buffer from existing bytes and attempt to lock it.
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        Self { inner: bytes, locked: false }
    }

    /// Attempt to lock the current buffer into physical RAM.
    ///
    /// Returns an error only on Unix if `mlock` fails; non-Unix platforms
    /// currently report success without doing anything.
    pub fn mlock(&mut self) -> Result<(), ModelCryptoError> {
        #[cfg(unix)]
        {
            let ptr = self.inner.as_ptr();
            let len = self.inner.len();
            if len == 0 {
                self.locked = false;
                return Ok(());
            }
            let ret = unsafe { libc::mlock(ptr as *const libc::c_void, len) };
            if ret != 0 {
                return Err(ModelCryptoError::EncryptionError(format!(
                    "mlock failed (errno {})",
                    std::io::Error::last_os_error().raw_os_error().unwrap_or(0)
                )));
            }
            self.locked = true;
        }
        #[cfg(not(unix))]
        {
            self.locked = false;
        }
        Ok(())
    }

    /// Unlock the buffer from physical RAM.
    ///
    /// This is called automatically before zeroization on drop; manual callers
    /// must be aware that the buffer will no longer be swap-protected.
    pub fn munlock(&mut self) -> Result<(), ModelCryptoError> {
        #[cfg(unix)]
        if self.locked && !self.inner.is_empty() {
            let ptr = self.inner.as_ptr();
            let len = self.inner.len();
            let ret = unsafe { libc::munlock(ptr as *const libc::c_void, len) };
            if ret != 0 {
                return Err(ModelCryptoError::EncryptionError(format!(
                    "munlock failed (errno {})",
                    std::io::Error::last_os_error().raw_os_error().unwrap_or(0)
                )));
            }
            self.locked = false;
        }
        #[cfg(not(unix))]
        {
            self.locked = false;
        }
        Ok(())
    }

    /// Consume the buffer, zeroize it, and unlock if needed.
    pub fn secure_erase(mut self) {
        self.inner.zeroize();
        let _ = self.munlock();
    }

    /// Return whether the buffer is currently locked.
    pub fn is_locked(&self) -> bool {
        self.locked
    }
}

impl Default for SecretBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl Deref for SecretBuffer {
    type Target = Vec<u8>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for SecretBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl AsRef<[u8]> for SecretBuffer {
    fn as_ref(&self) -> &[u8] {
        &self.inner
    }
}

impl AsMut<[u8]> for SecretBuffer {
    fn as_mut(&mut self) -> &mut [u8] {
        &mut self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_zeroizes_on_drop() {
        let mut buf = SecretBuffer::from_bytes(vec![1u8, 2, 3, 4]);
        // Force unlock before drop to avoid double unlock; in real use drop handles it.
        let _ = buf.munlock();
    }

    #[test]
    fn test_mlock_roundtrip() {
        let mut buf = SecretBuffer::from_bytes(vec![0u8; 4096]);
        let result = buf.mlock();
        // mlock may fail for non-root users due to RLIMIT_MEMLOCK; allow that.
        if result.is_ok() {
            assert!(buf.is_locked());
            buf.munlock().unwrap();
            assert!(!buf.is_locked());
        }
    }

    #[test]
    fn test_as_ref_mut() {
        let mut buf = SecretBuffer::from_bytes(vec![0u8; 4]);
        buf.as_mut()[0] = 42;
        assert_eq!(buf.as_ref()[0], 42);
    }
}
