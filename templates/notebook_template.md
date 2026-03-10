# Notebook Template

Use this structure for every curriculum notebook.

## 1. Title cell (Markdown)

- Notebook title
- One-paragraph overview
- Short note on what the learner should observe or listen for

## 2. Setup cell (Code)

```python
import sys
from pathlib import Path

ROOT = Path.cwd().resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from rf_utils import *
from IPython.display import Audio, Markdown, display
import ipywidgets as widgets
import matplotlib.pyplot as plt

%matplotlib widget
```

## 3. Concept cells (Markdown + Code)

- Explain one concept at a time
- Follow each explanation with a runnable code example
- Keep equations close to the code that demonstrates them

## 4. Interactive cell pattern

```python
fig, ax = plt.subplots(figsize=(10, 3))
audio_out = widgets.Output()

def update_example(freq=1000.0, amplitude=1.0):
    ax.clear()
    t, sig = generate_tone(freq=freq, duration=0.01, fs=44_100, amplitude=amplitude)
    ax.plot(t * 1e3, sig)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Interactive signal view")
    fig.canvas.draw_idle()

    with audio_out:
        audio_out.clear_output()
        display(Audio(sig, rate=44_100))

controls = widgets.interactive(
    update_example,
    freq=widgets.FloatSlider(min=100, max=4000, step=50, value=1000, description="Freq (Hz)"),
    amplitude=widgets.FloatSlider(min=0, max=1, step=0.05, value=0.8, description="Amplitude"),
)

display(controls, audio_out)
```

## 5. What to try cell (Markdown)

- Give 2-4 specific parameter changes to try
- State what should change in the waveform, spectrum, or audio

## 6. Key takeaway cell (Markdown)

- End with a short summary of the physical or DSP intuition the learner should keep
