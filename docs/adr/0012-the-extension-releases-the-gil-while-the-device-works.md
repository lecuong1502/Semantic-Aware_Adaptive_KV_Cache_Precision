# The extension releases the GIL while the device works

Milestone 1 puts Python threads beside the decode loop. The pressure monitor
(#45) polls NVML every 50 ms from a background thread, and the hold tool
(#51) writes its status four times a second from another. The spec assumed
that decoding does not block them, because ctypes releases the GIL during an
NVML call. That is only half of it: a thread must hold the GIL to run at all,
and the decode loop held it for seconds.

#51 measured this. At 8K positions on Qwen2.5-1.5B, the status thread went
unwritten for up to 14.7 s during prefill. Two things held the GIL while the
device worked:

- **Launches.** The `device` module's bindings held the GIL through every
  call. A launch blocks when the device's queue is full, and a copy back to
  the host waits for every kernel queued before it.
- **Frees.** Each step uploads its token ids and positions as a
  `DeviceIndex`, and Python frees them as the step ends, with the GIL held.
  `cudaFree` waits for every kernel queued before it, so one free held the
  GIL through a whole prefill chunk's work: 0.9 s for three chunks, measured.

## Decision

**Every binding that launches device work, copies to or from it, or frees
device memory per step, releases the GIL while it does.**

- Kernel launches, `logits`/`greedy` and the uploads release it around the
  device work and the copy.
- A `DeviceIndex` is freed without it (`FreeWithoutGil`).
- **No binding that releases the GIL takes a Python object by value.** A
  `Span` holds a reference to the tensor it views, and a `Span` parameter
  destroyed with the GIL released would drop that reference without it. The
  review of #51 found five `std::optional<Span>` parameters taken by value.
  Spans are now taken by const reference, and are destroyed by the pybind
  caster that loaded them, once the GIL is held again.

`tests/test_hold.py` guards the decision. A thread ticks every 5 ms beside a
4096-token prefill on Qwen2.5-0.5B, and no tick may be more than 100 ms late.
With the bindings as they were, a tick was 1.08 s late.

## Consequences

- The monitor's 50 ms poll holds while the engine prefills or decodes, as
  #45 assumed.
- A new binding that launches device work must release the GIL too. Frees
  outside the per-step path, such as a workspace or a cache released once,
  still hold it. Each waits for the work before it once, and has not been
  measured as a stall.
- Releasing the GIL does not make the extension thread-safe for concurrent
  device work. Only one thread drives the engine.
