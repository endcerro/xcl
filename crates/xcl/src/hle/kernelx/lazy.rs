//! Physical pages mapped into the title's own reservations, committed on first use.
//!
//! A title maps its physical pages wherever it reserved address space and places resources
//! there – render targets and depth buffers too, which on the PC live in video memory
//! (`d3d12x::vram`); the CPU never touches those pages. Committed at map time they doubled
//! that memory (a retail game: 5.4 GB of mapped pages next to its video-memory targets, past
//! the system's commit). So a page whose contents are still zero is mapped as reserved address
//! space only: the CPU's first access faults, the page is committed (zeros, as the page holds)
//! and the access goes on. Kernel calls do not fault in the title – they fail – so the buffers
//! of the file calls XCL routes, and DirectStorage's destinations, are committed first
//! (`ensure`).

use std::collections::BTreeSet;
use std::ffi::c_void;
use std::sync::{Once, RwLock};

use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, NTSTATUS};
use windows_sys::Win32::System::Diagnostics::Debug::{AddVectoredExceptionHandler, EXCEPTION_POINTERS};
use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, PAGE_READWRITE};

const PAGE: usize = 0x1_0000;
const EXCEPTION_CONTINUE_EXECUTION: i32 = -1;
const EXCEPTION_CONTINUE_SEARCH: i32 = 0;

/// pages (64 KiB) mapped but not committed yet
static LAZY: RwLock<BTreeSet<usize>> = RwLock::new(BTreeSet::new());

fn install() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(handler));
    });
}

/// Commit `page` if it is a lazy one (true: it was).
unsafe fn commit(page: usize) -> bool {
    let mut g = LAZY.write().unwrap_or_else(|e| e.into_inner());
    if !g.remove(&page) {
        return false;
    }
    !VirtualAlloc(page as *const c_void, PAGE, MEM_COMMIT, PAGE_READWRITE).is_null()
}

unsafe extern "system" fn handler(info: *mut EXCEPTION_POINTERS) -> i32 {
    let r = &*(*info).ExceptionRecord;
    if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS || r.NumberParameters < 2 {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    let at = r.ExceptionInformation[1];
    let page = at & !(PAGE - 1);
    let lazy = LAZY.read().unwrap_or_else(|e| e.into_inner()).contains(&page);
    if lazy && commit(page) {
        return EXCEPTION_CONTINUE_EXECUTION;
    }
    // another thread (or `ensure`) committed the page between this fault and its handling – it
    // left the lazy set first: a read or a write goes on (passed on, the title crashed on its
    // first write into a fresh page)
    if r.ExceptionInformation[0] <= 1 && read_write(at) {
        return EXCEPTION_CONTINUE_EXECUTION;
    }
    EXCEPTION_CONTINUE_SEARCH
}

/// Whether `a` is in committed read-write memory (what `commit` makes of a lazy page).
unsafe fn read_write(a: usize) -> bool {
    let mut m: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
    VirtualQuery(a as *const c_void, &mut m, std::mem::size_of_val(&m)) != 0 && m.State == MEM_COMMIT && m.Protect == PAGE_READWRITE
}

/// `page` (reserved, 64 KiB aligned) is mapped: committed on first use.
pub(crate) fn mark(page: usize) {
    install();
    LAZY.write().unwrap_or_else(|e| e.into_inner()).insert(page);
}

/// `page` is unmapped: true if it was never used (nothing to keep of it).
pub(super) fn unmark(page: usize) -> bool {
    LAZY.write().unwrap_or_else(|e| e.into_inner()).remove(&page)
}

/// The lazy pages of `[p, p + len)` committed – memory a kernel call or XCL writes.
pub fn ensure(p: usize, len: usize) {
    if len == 0 {
        return;
    }
    let pages: Vec<usize> = {
        let g = LAZY.read().unwrap_or_else(|e| e.into_inner());
        if g.is_empty() {
            return;
        }
        g.range(p & !(PAGE - 1)..p.saturating_add(len)).copied().collect()
    };
    for page in pages {
        unsafe { commit(page) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{VirtualFree, MEM_RELEASE, MEM_RESERVE};

    /// A lazy page reads as zeros on first touch, keeps what is written; `ensure` commits it
    /// without a fault.
    #[test]
    fn a_lazy_page_is_committed_on_first_use() {
        unsafe {
            let r = VirtualAlloc(std::ptr::null(), 2 * PAGE, MEM_RESERVE, PAGE_READWRITE) as usize;
            mark(r);
            mark(r + PAGE);
            assert_eq!(std::ptr::read_volatile((r + 8) as *const u32), 0);
            std::ptr::write_volatile((r + 8) as *mut u32, 7);
            assert_eq!(std::ptr::read_volatile((r + 8) as *const u32), 7);
            assert!(!unmark(r), "used: no longer lazy");
            ensure(r + PAGE + 4, 4);
            assert!(!LAZY.read().unwrap().contains(&(r + PAGE)));
            assert!(crate::gpu::shader::readable(r + PAGE, 4));
            VirtualFree(r as *mut c_void, 0, MEM_RELEASE);
        }
    }

    /// Threads touching a lazy page at once all go on: one of them commits it, and the faults
    /// of the others, handled once the page has left the lazy set, are retried rather than
    /// passed on (a retail game's worker crashed on its first write into a fresh page).
    #[test]
    fn concurrent_first_touches_of_a_lazy_page_all_go_on() {
        const THREADS: usize = 16;
        const ROUNDS: usize = 64;
        unsafe {
            let r = VirtualAlloc(std::ptr::null(), ROUNDS * PAGE, MEM_RESERVE, PAGE_READWRITE) as usize;
            assert_ne!(r, 0);
            for k in 0..ROUNDS {
                mark(r + k * PAGE);
            }
            let start = std::sync::Arc::new(std::sync::Barrier::new(THREADS));
            let threads: Vec<_> = (0..THREADS)
                .map(|t| {
                    let start = start.clone();
                    std::thread::spawn(move || {
                        for k in 0..ROUNDS {
                            start.wait();
                            std::ptr::write_volatile((r + k * PAGE + t * 64) as *mut u32, t as u32 + 1);
                        }
                    })
                })
                .collect();
            for t in threads {
                t.join().unwrap();
            }
            for k in 0..ROUNDS {
                for t in 0..THREADS {
                    assert_eq!(std::ptr::read_volatile((r + k * PAGE + t * 64) as *const u32), t as u32 + 1);
                }
            }
            VirtualFree(r as *mut c_void, 0, MEM_RELEASE);
        }
    }
}
