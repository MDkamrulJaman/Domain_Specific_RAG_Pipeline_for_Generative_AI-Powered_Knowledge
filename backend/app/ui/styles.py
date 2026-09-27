"""Visual tokens and presentation assets; no provider or request logic."""
from pathlib import Path
import gradio as gr

CSS = Path(__file__).with_name("theme.css").read_text(encoding="utf-8")
DARK_HEAD = "<script>document.documentElement.classList.add('dark');</script>"
INITIAL_THEME = "() => { document.documentElement.classList.add('dark'); document.body.classList.add('dark'); }"
TOGGLE_THEME = "(enabled) => { document.documentElement.classList.toggle('dark', enabled); document.body.classList.toggle('dark', enabled); }"

HEADER = """
<header class="hero">
  <div class="brand"><span class="brand-symbol" aria-hidden="true">✦</span> KNOWLEDGE / AI</div>
  <div class="hero-content">
    <div><p class="eyebrow">YOUR DOCUMENTS. A CLEARER PICTURE.</p>
    <h1>Less searching.<br><span>More understanding.</span></h1>
    <p class="hero-description">Turn technical documents into a conversation.<br>Choose your assistant and find the answers that matter.</p></div>
    <div class="hero-aside"><span class="step">01 <b>Choose a provider</b></span><span class="step">02 <b>Add your documents</b></span><span class="step">03 <b>Ask a question</b></span></div>
  </div>
</header>
"""
EMPTY_CHAT = """
<div class="chat-welcome"><div class="welcome-icon" aria-hidden="true">✧</div>
<h2>Your next insight starts here.</h2><p>Upload a document to the selected provider,<br>then ask a question in your own words.</p>
<div class="prompt-hints"><span>Summarize the key points</span><span>Explain a technical concept</span><span>Find a specific requirement</span></div></div>
"""


def build_theme():
    return gr.themes.Soft(
        primary_hue="teal", secondary_hue="cyan", neutral_hue="slate",
        font=["Inter", "Segoe UI", "sans-serif"],
    ).set(
        body_background_fill_dark="#090f1b",
        body_text_color_dark="#e2e8f0",
        block_background_fill_dark="#101a2b",
        block_border_color_dark="#263449",
        input_background_fill_dark="#0c1524",
        button_primary_background_fill_dark="#5eead4",
        button_primary_text_color_dark="#062c29",
        button_primary_background_fill_hover_dark="#99f6e4",
    )
