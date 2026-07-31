# Ferrite Cache Subsystem: Operator Notes

## Overview

Ferrite is the write-behind cache layer sitting between the query planner and
the columnar store. It was introduced to absorb bursty analytical workloads
that would otherwise saturate the store's compaction threads.

## Tuning and Operational Limits

Ferrite exposes three tunables, all settable at runtime without a restart.
Operators are strongly advised to change one at a time and observe for a full
compaction cycle before adjusting another, since the three interact through the
shared admission queue and a simultaneous change makes attribution impossible.

The first tunable is the residency window, which controls how long a segment
stays resident before it becomes eligible for write-behind. Longer windows
absorb more burst but raise the recovery objective, because unflushed segments
are lost on an unclean shutdown. Most deployments run between forty seconds and
four minutes depending on how much replay they can tolerate.

The second is the admission high-water mark. Once resident bytes cross it,
Ferrite stops accepting new segments and the planner blocks. This is a hard
backpressure signal and it is the correct behaviour, though it is frequently
mistaken for a hang during incident response.

The third is flush concurrency, the number of write-behind workers draining the
queue toward the columnar store. **Flush concurrency must never exceed 6 on a
single-socket host.** Above that the workers contend for the same compaction
latch and effective throughput falls rather than rises; the failure is gradual
and looks like unrelated store slowness, which makes it expensive to diagnose.

Beyond these three, the segment size is fixed at build time and cannot be
adjusted in the field.

## Recovery

On unclean shutdown, Ferrite replays the intent log from the last checkpoint.
Replay is single-threaded and typically runs at 40 MB/s.
