# Lab notes: running the AIME comparison on an RTX 4060

A record of moving this harness from the Mac to an 8 GB desktop GPU and running
the AIME 2025 sweep there, 11 to 18 September 2026. It lists what was done, what
went wrong, how each problem was found, and what should have been done first.
The point is the lessons, so the mistakes are written down as plainly as the
fixes, including the ones made while running the experiment.

## What should have been done before the long run

Most of the time lost below traces back to one of these. Next time, do them in
this order before starting anything that runs for hours.

1. **Run a full-length probe on the target machine.** One or two problems at the
   real token cap, with the real settings. It shows whether a full answer fits in
   memory, how long an answer takes, and what the model does at length. The
   two-problem probe here found the memory ceiling and the near-cap answer; it
   should also have been run with thinking off, which would have shown the
   looping before 30 answers were spent on it.
2. **Estimate the total time from measured numbers, not from the README.** The
   README assumed 4 to 6 answers generating at once. On 8 GB the real figure is
   1 for MiniCPM5 and about 1.5 for Qwen3.5. The first full estimate was about
   27 hours of GPU time.
3. **Fix every setting before the first answer is generated.** Attempts per
   problem, the token cap, loop handling and sampler settings all changed or were
   questioned partway through. Each mid-run change has to be disclosed and makes
   the stage less uniform.
4. **Test the grader on bad output, not only good output.** Truncated answers,
   answers ending inside `\boxed{`, nested braces, no box at all. The grader bug
   found here only shows up on answers cut off at the cap.
5. **Save the full text of every answer.** Only the last 400 characters were
   kept, so 4 answers could not be re-graded after the grader was fixed.
6. **Save each answer as it finishes, and force it to disk.** Two power cuts in
   one day each lost a whole batch before this was in place.
7. **Check the vendor's recommended settings for each mode, then decide and
   write down whether to follow them.** Qwen recommends different sampling for
   thinking-off mode than the harness uses for every model. Using one sampler for
   all models is the fair choice, but it has to be a stated decision, not
   something discovered from the results.
8. **Test every safety and monitoring tool by forcing it to trigger.** The heat
   watchdog was tested this way and worked. Two monitors that were not tested
   this way turned out to be blind.
9. **Make any failure stop the pipeline.** A crashed stage once let the runner
   move on to the next stage.
10. **Keep a UPS on the machine,** or accept that every power cut costs the
    answers in progress.

## Timeline

| When | What happened |
|---|---|
| 11 Sep | Cloned the repo on the Arch desktop. Fixed `fetch.sh` for Linux, built a Python 3.12 environment, got vLLM running, found the memory limits. |
| 18 Sep 00:41 | Two-problem MiniCPM5 probe at the full 32k cap. 1 of 2 correct, peak 67°C, one answer used 29,114 of 32,768 tokens. |
| 18 Sep 01:04 | Full sweep started: Qwen3.5-2B thinking off, 30 problems x 4 attempts. |
| 18 Sep ~02:50 | Paused at 30/120 for a shutdown. Code committed. 15 of 30 answers had hit the cap. |
| 18 Sep 13:40 | Resumed. Checked the cut-off answers first: all 15 were loops, not long reasoning. |
| 18 Sep ~14:06 | Power cut. The batch in progress was lost. The results file survived; the GPU log got NUL bytes. |
| 18 Sep 14:17 | Resumed after cleaning the log and hardening `progress.py`. |
| 18 Sep 14:55 | Switched to saving each answer as it finishes, at 36/120. |
| 18 Sep 15:22 | Switched to 2 attempts per problem, loop stopping and a fixed grader, at 39 answers. First restart crashed on memory and the runner advanced to the next stage. Stopped it, fixed the runner. |
| 18 Sep 15:24 | Resumed with all changes. |
| 18 Sep 16:13 | Stage 1 finished, 79 attempts, avg@n 17.5. |
| 18 Sep 18:03 | Stage 2 finished, avg@n 11.7. 45 of 60 attempts stopped as loops; the 7 that terminated were all correct. |
| 18 Sep 18:04 | Stage 3 (MiniCPM5) started. Third power cut at ~18:40; per-answer saving held, 12 answers kept. |
| 18 Sep 18:59 | Out of memory mid-run: started at 0.92 unlocked, hyprlock took 169 MiB when the screen locked. Runner fixed at 0.90 with auto-resume, up to 5 OOMs per stage. |
| 18 Sep 20:54 | **Fourth power cut**, 7 answers into the stage. Machine stayed off for a week. |
| 25 Sep 16:42 | Resumed. 24 answers had survived; only the one in flight was lost. Both Qwen stages replayed in ~1 min each and exited 0 with saved work skipped, confirming them complete. |
| 25 Sep 21:16 | Stage 3 finished, 60 attempts, avg@n 71.7. Sweep complete as originally designed. |
| 27 Sep 17:07 | Stage 4 added: Qwen thinking on with Qwen's own sampler (`top_k 20`, `presence_penalty 1.5`), to test whether the shared sampler was responsible for the loops. |
| 27 Sep 20:48 | Stage 4 finished, avg@n 15.0. Paired gain over the shared sampler +3.3 points, CI touching zero. |
| 28 Sep | Written up in `FINDINGS.md`. Branch pushed. |

## Problems found

Each entry says what happened, how it was noticed, what was done, and the lesson.

### Setting up the machine

**`fetch.sh` only worked on a Mac.** It read file sizes with BSD `stat -f%z`,
which fails on Linux. A rerun also printed a zsh "no matches found" error on a
glob. Fixed by detecting which `stat` is available and using a zsh glob
qualifier. *Lesson: a script written on one OS has not been tested on another.*

**System Python was too new.** Arch ships Python 3.14; vLLM does not support it.
Built the environment with `uv venv --python 3.12`. *Lesson: pin the Python
version in the setup instructions.*

**vLLM could not build its GPU code.** Three separate failures, each hidden
behind the previous one:
1. `nvcc` not found. The pip packages ship a CUDA compiler inside the venv but
   do not put it on `PATH`.
2. `ninja` not found, for the same reason.
3. flashinfer's sampling kernel will not compile: its bundled headers do not
   match the pip CUDA 13.4 compiler.

Fixed in `backends.py` by pointing `CUDA_HOME` at the venv's CUDA, adding it to
`PATH`, and disabling flashinfer's sampler. That last change swaps which code
does the sampling, so it is printed at startup rather than applied silently.
*Lesson: read the root cause in the log, not the last line. Each error was
several screens above the final "Engine core initialization failed".*

**The vLLM path ignored the downloaded model** and would have downloaded it
again, contradicting the function's own docstring. Fixed in `resolve_path`.

**`smoke.py` crashed on Qwen.** Qwen3.5's image processor starts CUDA in the
main process, which forces vLLM to start workers by re-importing the script.
The script had no `if __name__ == "__main__"` guard. Fixed.

### Hardware limits

**Memory, not heat, is the real limit on an 8 GB card.** Measured, with the
model already loaded:

| Setting | Space for answer text, MiniCPM5-2B |
|---|---|
| Default, CUDA graphs on | 12,928 tokens |
| `--eager`, CUDA graphs off | 40,432 tokens |

A full AIME answer can reach 32,768 tokens, so it only fits with `--eager`, one
at a time. Qwen3.5-2B fits about 1.5 at once, not the ~3.5 the README's
per-token estimate implied, because its linear-attention layers need memory too.

**The desktop takes a changing share of GPU memory.** About 350 MiB normally.
The screen locker, hyprlock, adds about 170 MiB while the screen is locked.
vLLM refuses to start unless its whole share is free, so a restart with the
screen locked failed at 0.92. Measured: 0.90 still holds a full-length answer
for both models, 0.88 does not. The runner now picks the setting at each start
and steps down on that specific refusal. Its first version estimated free
memory in a separate process and was about 0.2 GiB too optimistic, which is how
the 15:22 restart failed. *Lesson: when a limit is enforced by another program,
test against that program's own check, not your estimate of it.*

**Heat was never a problem.** Peak 71°C over the whole run, against a throttle
point near 83°C. The card is also hard-limited to 115 W. The watchdog stops the
run at 85°C; it was tested by setting the limit to 1°C, and it stopped the model
and freed the GPU within 20 seconds.

### Measuring and grading

**The chat-template finding held up on the real files.** Rendering MiniCPM5's
template through the real tokenizer gave the same three states the auditor
predicted, and generation behaved accordingly with thinking on and off.

**Qwen3.5-2B with thinking off loops.** 22 of its first 38 answers ran to the
32,768-token cap, each taking 12 to 15 minutes. Every one was a loop: most
repeated their final `\boxed{}` answer forever, a few repeated a line of
reasoning. The harness was checked first: stop tokens work, and the answers
that ended normally went up to 17,751 tokens. So the model reaches an answer
and then fails to stop. *Lesson: when a number looks wrong, read the raw
outputs before changing any settings. Raising the cap would only have made
these loops longer.*

**The grader misread answers cut off at the cap.** An answer looping on
`\boxed{16}` that was cut off at `\boxed{1` was graded as 1. It took the last
`\boxed` even when it was unfinished. 4 of the 22 capped answers were
misgraded, and one of them was actually correct. Fixed to take the last
complete box. The saved answers were re-graded from their saved endings, with
the original file kept as `*.before-rescore.jsonl`. 4 answers had no complete
box in the 400 characters that were saved, so they could not be checked and
are flagged `pred_uncertain`. *Lesson: see items 4 and 5 of the checklist.*

**Saved speed figures were wrong for batched runs.** Batch time was split
evenly across answers, so a short answer showed 7.6 tokens per second and a
long one 86.7. Since per-answer saving, each answer records its own time from
start to finish. The two methods are not comparable, and neither is a clean
cost per answer.

**The sampler is not the one Qwen recommends or evaluates with.** The harness
uses temperature 0.6 and top_p 0.95 for every model, with no top_k and no
presence penalty. Qwen's Qwen3.5-2B model card, read only on 18 September after
the thinking-on stage started looping too, says:

- The model "is more prone to entering thinking loops compared to other
  Qwen3.5 models", and recommends streaming generation to detect and interrupt
  them.
- Qwen's own benchmarks used top_k 20 and a presence penalty of 1.5 in both
  modes: temperature 0.6 and top_p 0.95 with thinking, temperature 0.7 and
  top_p 0.8 without.

The presence penalty in particular discourages repetition. Using one sampler
for every model is a defensible choice for a fair comparison, but here it
probably costs Qwen points. It must be stated in any write-up. *Lesson: read the
model card's generation settings before the first run. It is item 7 of the
checklist, and it would have predicted the looping.*

### Surviving interruptions

**Power cuts lost whole batches.** Answers were saved only when all 6 in a
batch finished. Now each answer is saved and forced to disk as it finishes, and
a new problem starts whenever one finishes, which also removed the wait for the
slowest answer in each batch.

**A power cut can corrupt the end of a file.** Reading now drops a half-written
last line, and appending first cuts it off so the next record is not glued to
it. The GPU log came back with NUL bytes after the second cut, which crashed
the progress display; it now skips damaged lines.

**A failed stage let the sweep move on.** The runner only stopped for the
watchdog. After a crash it started the next stage. Caught within a minute,
before that stage saved anything. Any failure now stops the sweep. Both paths
were tested with a fake Python before being trusted.

### Monitoring mistakes

These were errors in the tools built to watch the run, not in the harness.

- **A temperature alert that could not see anything.** `nvidia-smi -f` buffers
  its log until it exits, so the file stayed empty while the alert watched it.
  Replaced with a loop that queries the card directly.
- **Process checks that matched themselves.** Twice, a watcher searched for
  running processes by name, and its own command line contained that name. One
  never ended; the other stopped itself after doing its job. Fixed by anchoring
  the search to the start of the command line.
- **A resume check that read the previous run's log.** It reported success
  immediately because the old log already contained the line it was waiting
  for. The real restart had failed.
- **An ETA that collapsed after a restart.** It measured time from the latest
  start but counted results from every session. Now it adds up all sessions and
  leaves out gaps.

*Lesson: a monitor you have not seen fire is not a monitor.*

## Changes made partway through, to disclose

The Qwen3.5-2B thinking-off stage is not uniform. Any write-up should say:

- **Attempts per problem dropped from 4 to 2** at 39 answers, to halve the run.
  The first problems keep 4 attempts. The official score averages within each
  problem first, so every problem counts once.
- **Loop stopping was added** at the same point. An answer is stopped once its
  last 3,000 characters are one block repeated exactly. It applies to the other
  two stages from their start. When the loop repeats a final answer, stopping
  cannot change the grade. When it is a reasoning loop with no answer yet,
  stopping assumes the model would not have broken out of it. That is likely
  but was not verified before the change; the earlier notes and commit message
  said it "cannot change" the answer, which is only true for the first kind.
  Every stopped answer is marked `looped`, so they can be re-run without the
  stop later to measure how often a loop would have recovered.
- **The grader was fixed and the saved answers re-graded.** One answer changed
  from wrong to correct; four are flagged as unverifiable.
- **Per-answer saving** changed what the saved speed figures mean, at 36
  answers. Scores are unaffected.

## Final state

28 September 2026. Sweep complete, four stages, 259 attempts, 17 hours of
wall-clock generation. Results and conclusions are in
[`FINDINGS.md`](FINDINGS.md); this file is the record of how the run went.

| Stage | Attempts | avg@n | Card |
|---|---|---|---|
| Qwen3.5-2B, thinking off | 79 | 17.5% | 29.6 |
| Qwen3.5-2B, thinking on | 60 | 11.7% | 29.6 |
| MiniCPM5-2B, thinking on | 60 | 71.7% | 86.5 |
| Qwen3.5-2B, Qwen's own sampler | 60 | 15.0% | 29.6 |

Both hypotheses were refuted: the chat template does not explain the card's
Qwen baseline, and neither does the sampler. The card's comparison holds.

The prediction in the 18 September version of this section — that the early
Qwen score being below the card's "fits the idea that the scorecard ran Qwen
with thinking off" — was wrong. Thinking on scored *lower*, not higher. It is
left here rather than edited out, because reading a partial result as support
for the hypothesis that motivated the run is the mistake this file exists to
record.

## Lessons added after 18 September

**A power cut you have already survived is not a power cut you have solved.**
Four cuts in this run. Per-answer fsync, added after the second, meant the
fourth cost one answer instead of a batch — and the week the machine then spent
off cost nothing at all. The fix was cheap and should have been there from the
first hour.

**Wait on a condition only the new run can produce.** Stale-log checks gave a
false reading twice during the run, and then a third time in a different form:
`progress.py` decided the sweep was finished by searching the whole of
`sweep.log` for `sweep complete`, a string that never goes away. Adding stage 4
to an already-complete sweep made it print `FINISHED` next to its own
`eval process running` and `GPU busy 100%` lines. Fixed to read only the last
event. The reason this stayed cosmetic rather than costing a run is that the
health block reports process, GPU and log age independently, so the wrong line
was contradicted by the three directly above it.

**Interim scores from a partial stage are biased, not just noisy.** Short
answers finish first. Stage 4 read 22.5% at 40 of 60 attempts and finished at
15.0%: the last 20 attempts produced no correct answers. A number quoted from a
running stage should carry the direction of its bias, not only a caution.

**Do not sum a per-answer timing field across a batched stage.** `seconds` is
measured from submission to finish, so with 6 answers in flight it is latency,
not exclusive GPU time. Summing it reported 46 hours of GPU time against 17
hours of actual wall clock, and made Qwen's cost per correct answer look 5x
worse than it was. Stage cost now comes from `sweep.log` wall clock.

**A decision made to save time still has to be tested.** Using one sampler for
every model was the right call for comparability, but it put Qwen in the
configuration its own card warns against. That was noticed from the results
rather than from the vendor documentation, and it took a fourth stage to find
out what it cost. Point 7 at the top of this file was written before that
stage existed; the stage is what turned it from a caveat into a measurement.

## Resuming

```bash
cd ~/Downloads/claude/chat-template-auditor
setsid nohup ./run_aime_4060.sh >> results/vllm-4060/sweep.log 2>&1 &
watch -n 30 .venv/bin/python progress.py
```

Re-running skips every answer already on disk. `progress.py` reads only saved
files and never touches the GPU, so it is safe to run at any time.
