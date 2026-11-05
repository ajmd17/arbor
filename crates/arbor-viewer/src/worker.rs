//! Work too slow for a frame, done off the UI thread: regrowing the tree, and baking its
//! leaf atlas, which runs to seconds for a clustered species.
//!
//! The latest request wins. One made while another is running replaces any still
//! waiting rather than queueing behind it, so dragging a slider never builds up a
//! backlog of trees nobody will see: the worker finishes the one in hand, then grows
//! wherever the slider is by then. Each result is shown as it lands, so the tree keeps
//! up with the slider as closely as the growing allows.
//!
//! The web has no thread to spare, so there a request is simply run on the next poll,
//! on the UI thread, as the viewer always did.

#[cfg(not(target_arch = "wasm32"))]
use std::sync::mpsc::{Receiver, channel};
use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::{Condvar, Mutex};

type Run<J, R> = Arc<dyn Fn(J) -> R + Send + Sync>;

pub struct Worker<J, R> {
    run: Run<J, R>,
    /// Numbers every request, so a result can be told from one already superseded.
    requested: u64,
    /// The newest request whose result has been handed back.
    delivered: u64,
    #[cfg(not(target_arch = "wasm32"))]
    shared: Arc<Shared<J>>,
    #[cfg(not(target_arch = "wasm32"))]
    results: Receiver<(u64, R)>,
    #[cfg(target_arch = "wasm32")]
    pending: Option<(u64, J)>,
}

#[cfg(not(target_arch = "wasm32"))]
struct Shared<J> {
    /// The one request waiting to be started; a newer one replaces it.
    slot: Mutex<Slot<J>>,
    wake: Condvar,
}

#[cfg(not(target_arch = "wasm32"))]
struct Slot<J> {
    job: Option<(u64, J)>,
    closed: bool,
}

impl<J: Send + 'static, R: Send + 'static> Worker<J, R> {
    /// A worker running `run` on its own thread. `name` is the thread's, for debuggers.
    pub fn new(name: &str, run: impl Fn(J) -> R + Send + Sync + 'static) -> Self {
        let run: Run<J, R> = Arc::new(run);
        #[cfg(not(target_arch = "wasm32"))]
        {
            let shared = Arc::new(Shared {
                slot: Mutex::new(Slot { job: None, closed: false }),
                wake: Condvar::new(),
            });
            let (tx, results) = channel();
            let (theirs, work) = (Arc::clone(&shared), Arc::clone(&run));
            std::thread::Builder::new()
                .name(name.to_string())
                .spawn(move || {
                    loop {
                        let (id, job) = {
                            let mut slot = theirs.slot.lock().unwrap();
                            loop {
                                if slot.closed {
                                    return;
                                }
                                if let Some(job) = slot.job.take() {
                                    break job;
                                }
                                slot = theirs.wake.wait(slot).unwrap();
                            }
                        };
                        if tx.send((id, work(job))).is_err() {
                            return;
                        }
                    }
                })
                .expect("spawn worker thread");
            Self { run, requested: 0, delivered: 0, shared, results }
        }
        #[cfg(target_arch = "wasm32")]
        {
            let _ = name;
            Self { run, requested: 0, delivered: 0, pending: None }
        }
    }

    /// Asks for `job` to be done, in place of any request not yet started.
    pub fn request(&mut self, job: J) {
        self.requested += 1;
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.shared.slot.lock().unwrap().job = Some((self.requested, job));
            self.shared.wake.notify_one();
        }
        #[cfg(target_arch = "wasm32")]
        {
            self.pending = Some((self.requested, job));
        }
    }

    /// Does `job` here and now, and drops whatever was asked for before it, so nothing
    /// older can land on top of it afterwards. For a capture, which has to have the
    /// result in hand before it takes its frame.
    pub fn run_now(&mut self, job: J) -> R {
        self.requested += 1;
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.shared.slot.lock().unwrap().job = None;
        }
        #[cfg(target_arch = "wasm32")]
        {
            self.pending = None;
        }
        self.delivered = self.requested;
        (self.run)(job)
    }

    /// The newest result in since the last poll, if it is newer than anything handed
    /// back already. Older ones that arrived in the meantime are dropped.
    pub fn poll(&mut self) -> Option<R> {
        let mut newest = None;
        #[cfg(not(target_arch = "wasm32"))]
        while let Ok((id, result)) = self.results.try_recv() {
            if id > self.delivered {
                self.delivered = id;
                newest = Some(result);
            }
        }
        #[cfg(target_arch = "wasm32")]
        if let Some((id, job)) = self.pending.take() {
            self.delivered = id;
            newest = Some((self.run)(job));
        }
        newest
    }

    /// Whether something asked for has yet to come back.
    pub fn busy(&self) -> bool {
        self.delivered < self.requested
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl<J, R> Drop for Worker<J, R> {
    fn drop(&mut self) {
        self.shared.slot.lock().unwrap().closed = true;
        self.shared.wake.notify_one();
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    fn wait<J: Send + 'static, R: Send + 'static>(w: &mut Worker<J, R>) -> R {
        let start = std::time::Instant::now();
        loop {
            if let Some(r) = w.poll() {
                return r;
            }
            assert!(start.elapsed().as_secs() < 10, "worker never answered");
            std::thread::yield_now();
        }
    }

    #[test]
    fn hands_back_what_was_asked_for() {
        let mut w = Worker::new("test", |x: u32| x * 2);
        assert!(!w.busy());
        w.request(21);
        assert!(w.busy());
        assert_eq!(wait(&mut w), 42);
        assert!(!w.busy());
    }

    #[test]
    fn the_latest_request_wins() {
        // The first job holds the thread while the rest arrive, so all but the last
        // are replaced before they start.
        let gate = Arc::new(Mutex::new(()));
        let held = gate.lock().unwrap();
        let theirs = Arc::clone(&gate);
        let mut w = Worker::new("test", move |x: u32| {
            drop(theirs.lock().unwrap());
            x
        });
        w.request(1);
        std::thread::sleep(std::time::Duration::from_millis(50));
        for x in 2..=5 {
            w.request(x);
        }
        drop(held);
        let mut seen = vec![wait(&mut w)];
        while w.busy() {
            seen.push(wait(&mut w));
        }
        assert_eq!(seen.last(), Some(&5));
        assert!(seen.len() <= 2, "superseded requests were run: {seen:?}");
    }

    #[test]
    fn nothing_older_lands_after_run_now() {
        let mut w = Worker::new("test", |x: u32| {
            std::thread::sleep(std::time::Duration::from_millis(30));
            x
        });
        w.request(1);
        assert_eq!(w.run_now(7), 7);
        std::thread::sleep(std::time::Duration::from_millis(100));
        assert_eq!(w.poll(), None);
        assert!(!w.busy());
    }
}
