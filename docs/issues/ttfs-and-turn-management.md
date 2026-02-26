# Reducing TTFS & Managing Turns in Full-Duplex Mode

## 1. About TTFS (Time To First Speech)

Thanks for the feedback! The TTFS doesn't vary much across different GPUs — it's primarily determined by the model's architecture, not compute speed.

### Why ~1.5s?

The current architecture uses **1-second audio units**. The TTS vocoder needs to accumulate enough text tokens (currently ~25 tokens = 1 second of audio) before it can produce the first audio chunk:

```
Text LLM generates tokens:  [tok1] [tok2] ... [tok25]  → 1s worth of text
                                                          ↓
TTS vocoder processes:                              [audio chunk 1] → first sound
                                                          ↓
                                              TTFS ≈ LLM time + TTS time ≈ 1.5s
```

### Our plan: 0.5s units

We are working on reducing the unit size from 1s to 0.5s, which should roughly halve the TTFS:

```
Current (1s units):     [=== 25 tokens ===] → TTS → 🔊  TTFS ≈ 1.5s
Planned (0.5s units):   [= 12 tokens =] → TTS → 🔊      TTFS ≈ 0.7-0.8s
```

We chose to keep 1s units initially to preserve model intelligence — smaller units can slightly degrade generation quality. The 0.5s version is a sweet spot we're targeting.

## 2. About Full-Duplex and Turns

Great observation! Let me clarify the duplex design:

### What "full-duplex" means here

The key distinction isn't "no turns" — it's that **input and output overlap**. The model considers real-time input while generating its response:

```
Half-duplex (Turn-based):
  User speaks:    [=========]
  Model replies:                [=========]
                  (no overlap)

Full-duplex:
  User speaks:    [=========]----[===]--------[==]---
  Model replies:       [============]---[========]---
                       ↑ overlap: model hears user while speaking
```

Even in full-duplex mode, there are still **turns** — the model decides when to start and stop speaking. But it can react to user input mid-turn.

### The turn structure

To get best speech generation quality, at the start of each turn, text generation runs slightly ahead of audio. Otherwise, the model feels uncertain about what it should say in the next 1s unit.

This may explain why you observed that response in the middle of turn is faster than response in the beginning of turn.

## 3. Workarounds You Can Try Now

### Method A: Make the first chunk speak more text (reduce perceived TTFS)

In `MiniCPMO45/modeling_minicpmo_unified.py`, around line 4665:

```python
# Current code (allows < 1s audio for the first chunk):
if self.tts_text_start_pos == 0:  # start of turn
    min_token_per_chunk = 0       # allows decoding < 1s audio
    force_flush = True
```

Change `min_token_per_chunk` to `25 + 1` to force the first chunk to accumulate a full 1s of audio before speaking:

```python
if self.tts_text_start_pos == 0:
    min_token_per_chunk = 25 + 1  # wait for full 1s of audio tokens
    force_flush = True
```

This can let you get the first speech faster, but the quality may be slightly degraded, for example, the model may **become hesitant to speak.**

### Method B: Suppress turn-end to keep the model in one long turn

You can extend turn duration by making it harder for the model to end a turn:

1. **Set `length_penalty` to 1.3–1.5** in your generation config. This penalizes the EOS token, making the model generate longer responses without breaking into a new turn.

2. Combined with Method A, this effectively keeps the model in a continuous speaking mode with minimal interruptions.

This may include some unwanted side effects, for example, the model may **become out of distribution.** You may also observe that the model not respond to new user input in the middle of a turn when the turn is too long.

### Example config

```python
# In your streaming_generate or chat config:
length_penalty = 1.3  # try 1.3 ~ 1.5, higher = longer turns
```

In the Turn-based Chat frontend, you can set this via the `len_pen` parameter in the input options.

---

These are interim solutions. The fundamental fix (0.5s units) is on our roadmap, and the next model release will include it.