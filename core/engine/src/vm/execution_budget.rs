use std::future::{Future, poll_fn};
use std::sync::atomic::{AtomicBool, AtomicU16, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::task::{Poll, Waker};
use std::time::Duration;

#[derive(Debug, Default)]
pub(crate) struct ExecutionBudget {
    duration: RwLock<Option<Duration>>,
    active: RwLock<Option<Arc<Deadline>>>,
    checks_remaining: AtomicU16,
}

#[derive(Debug, Default)]
struct Deadline {
    expired: AtomicBool,
    cancelled: AtomicBool,
    waker: Mutex<Option<Waker>>,
}

impl ExecutionBudget {
    pub(crate) fn set_duration(&self, duration: Option<Duration>) {
        assert!(
            duration.is_none_or(|duration| !duration.is_zero()),
            "the execution time limit must be positive"
        );
        *self.duration.write().expect("execution duration poisoned") = duration;
    }

    pub(crate) fn begin(self: &Arc<Self>) -> ExecutionScope {
        let schedule = {
            let mut active = self.active.write().expect("execution budget poisoned");
            let duration = *self.duration.read().expect("execution duration poisoned");
            if active.is_none() {
                duration.map(|duration| {
                    let deadline = Arc::new(Deadline::default());
                    *active = Some(Arc::clone(&deadline));
                    self.checks_remaining.store(0, Ordering::Relaxed);
                    (deadline, duration)
                })
            } else {
                None
            }
        };
        let owns_deadline = schedule.is_some();
        if let Some((deadline, duration)) = schedule {
            drop(nagoya::spawn(async move {
                let mut sleep = std::pin::pin!(nagoya::sleep(duration));
                let elapsed = poll_fn(|cx| {
                    if deadline.cancelled.load(Ordering::Acquire) {
                        return Poll::Ready(false);
                    }
                    *deadline.waker.lock().expect("execution waker poisoned") =
                        Some(cx.waker().clone());
                    if deadline.cancelled.load(Ordering::Acquire) {
                        return Poll::Ready(false);
                    }
                    sleep.as_mut().poll(cx).map(|()| true)
                })
                .await;
                if elapsed && !deadline.cancelled.load(Ordering::Acquire) {
                    deadline.expired.store(true, Ordering::Release);
                }
            }));
        }
        ExecutionScope {
            budget: Arc::clone(self),
            owns_deadline,
        }
    }

    pub(crate) fn expired(&self) -> bool {
        self.active
            .read()
            .expect("execution budget poisoned")
            .as_ref()
            .is_some_and(|deadline| deadline.expired.load(Ordering::Acquire))
    }

    #[inline]
    pub(crate) fn should_interrupt(&self) -> bool {
        let remaining = self.checks_remaining.load(Ordering::Relaxed);
        if remaining != 0 {
            self.checks_remaining
                .store(remaining - 1, Ordering::Relaxed);
            return false;
        }
        self.checks_remaining.store(1023, Ordering::Relaxed);
        self.expired()
    }
}

/// Nested host and VM entries share the first scope's cancellation signal.
#[derive(Debug)]
#[must_use = "retain the scope for the execution boundary"]
pub struct ExecutionScope {
    budget: Arc<ExecutionBudget>,
    owns_deadline: bool,
}

impl Drop for ExecutionScope {
    fn drop(&mut self) {
        if !self.owns_deadline {
            return;
        }
        let deadline = self
            .budget
            .active
            .write()
            .expect("execution budget poisoned")
            .take();
        if let Some(deadline) = deadline {
            deadline.cancelled.store(true, Ordering::Release);
            let waker = deadline
                .waker
                .lock()
                .expect("execution waker poisoned")
                .take();
            if let Some(waker) = waker {
                waker.wake();
            }
        }
    }
}
