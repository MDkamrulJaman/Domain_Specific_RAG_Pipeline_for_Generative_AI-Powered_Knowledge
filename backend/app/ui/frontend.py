"""Gradio layout and event wiring. Request logic lives in handlers.py."""
import gradio as gr
from app.core.config import AppSettings
from app.services.skills import SKILLS
from app.services.provider_service import provider_configuration
from app.ui.handlers import status_markdown, provider_panel, refresh_provider, rag_answer, ingest_file
from app.ui.styles import CSS, HEADER, EMPTY_CHAT, INITIAL_THEME, TOGGLE_THEME, DARK_HEAD, build_theme


def create_demo():
    default = provider_configuration("pinecone")
    limits = AppSettings()
    with gr.Blocks(title="Generative AI-Powered Retrieval System for Technical Documentation⁠", analytics_enabled=False, delete_cache=(3600, 86400)) as demo:
        receipts = gr.State({"shared": []})
        gr.HTML(HEADER)
        with gr.Row(elem_id="toolbar"):
            gr.Markdown("**WORKSPACE** / Technical knowledge", elem_id="breadcrumb")
            dark_mode = gr.Checkbox(value=True, label="Dark mode", scale=0, min_width=130)
        dark_mode.change(fn=None, inputs=dark_mode, outputs=None, js=TOGGLE_THEME, queue=False)
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=300, elem_id="workspace-sidebar"):
                gr.Markdown("### Your workspace\nChoose where to upload and ask.")
                provider = gr.Dropdown(
                    choices=[("NVIDIA model", "nvidia"), ("Pinecone Assistant", "pinecone")],
                    value="pinecone", label="Answer provider",
                )
                connection_status = gr.Markdown(status_markdown(default), elem_id="provider-status")
                refresh = gr.Button("Refresh connection & files")
                top_k = gr.Dropdown(
                    choices=[("Automatic (based on skill)", "auto")] + [(str(n), str(n)) for n in range(1, 21)],
                    value="auto", label="NVIDIA retrieval depth", visible=False,
                    info="Automatic: General 5, Explain 4, Summarize/Requirements 8 passages. Manual values override this. Assistant manages its own retrieval.",
                )
                with gr.Group(visible=False) as nvidia_options:
                    skill = gr.Dropdown(
                        choices=[(policy.label, name) for name, policy in SKILLS.items()],
                        value="general", label="NVIDIA skill",
                        info="Applies to your next question. Summaries cover retrieved passages only. Use General answer for longer responses.",
                    )
                    thinking = gr.Checkbox(value=default["enable_thinking"], label="Enable NVIDIA thinking", info="Off for faster answers. On for additional reasoning.")
                gr.Markdown("### Document library")
                library_note = gr.Markdown("Files upload to Pinecone Assistant. Refresh to check when processing is complete.")
                upload = gr.File(label=f"PDF or TXT · up to {limits.MAX_UPLOAD_MB} MB", elem_id="document-upload", file_types=[".pdf", ".txt"], type="filepath")
                upload_button = gr.Button("Upload to Pinecone Assistant", variant="primary")
                upload_status = gr.Markdown("Files upload only to Pinecone Assistant. Wait until their status is Available before chatting.")
                library = gr.Dataframe(headers=["Document", "Status"], datatype=["str", "str"], value=[], interactive=False, label="Your documents", elem_id="document-library")
            with gr.Column(scale=3, min_width=320, elem_id="conversation-panel"):
                gr.Markdown("### Ask your documents\nClear answers, grounded in your knowledge.")
                chatbot = gr.Chatbot(label="Document answers", height=540, placeholder=EMPTY_CHAT, elem_id="knowledge-chat")
                response_status = gr.Markdown("Pinecone Assistant - Ready" if default["configured"] else "Pinecone Assistant - Setup required", elem_id="response-status")
                chat_interface = gr.ChatInterface(
                    fn=rag_answer, chatbot=chatbot,
                    additional_inputs=[provider, top_k, thinking, skill],
                    additional_outputs=[response_status],
                    # Message-only examples retain the current provider and controls.
                    examples=[["Summarize the key points"], ["Explain a technical concept"],
                              ["Find a specific requirement"]],
                    run_examples_on_click=True, cache_examples=False,
                    submit_btn="Send", stop_btn="Stop", analytics_enabled=False,
                    concurrency_limit=limits.UI_CONCURRENCY,
                )
                gr.Markdown("NVIDIA retrieves from the Pinecone index; Assistant retrieves from its own uploaded files. Each question is independent.")
                gr.HTML('<a href="/docs" target="_blank">API documentation</a> · <a href="/health" target="_blank">System health</a>')
        # Event wiring coordinates views; Gradio supplies event dispatch.
        provider.change(
            provider_panel, inputs=[provider, receipts],
            outputs=[connection_status, nvidia_options, thinking, upload_button, library, library_note, upload_status, response_status, top_k],
        )
        chat_interface.textbox.stop(
            lambda: "Response stopped. Partial answer kept.", outputs=response_status, queue=False,
        )
        chatbot.clear(lambda: "Ready for a new question.", outputs=response_status, queue=False)
        # Serialize library updates while an upload or refresh is running.
        busy_controls = [provider, refresh, upload_button]
        def lock_controls():
            return [gr.update(interactive=False) for _ in busy_controls]
        def unlock_controls():
            return [gr.update(interactive=True) for _ in busy_controls]
        refresh.click(lock_controls, outputs=busy_controls, queue=False).then(
            refresh_provider, inputs=[provider, receipts], outputs=[connection_status, receipts, library],
        ).then(unlock_controls, outputs=busy_controls, queue=False)
        upload_button.click(lock_controls, outputs=busy_controls, queue=False).then(
            ingest_file, inputs=[upload, receipts, provider], outputs=[upload_status, receipts, library],
        ).then(unlock_controls, outputs=busy_controls, queue=False)
    demo.queue(max_size=limits.UI_QUEUE_SIZE, default_concurrency_limit=limits.UI_CONCURRENCY)
    return demo


def mount_demo(app, demo):
    return gr.mount_gradio_app(
        app, demo, path="/", css=CSS, js=INITIAL_THEME, head=DARK_HEAD,
        max_file_size=AppSettings().max_upload_bytes, show_error=False, footer_links=[],
        theme=build_theme(),
    )
