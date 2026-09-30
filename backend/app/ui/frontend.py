"""Gradio layout and event wiring. Request logic lives in handlers.py."""
import gradio as gr
from app.core.config import AppSettings
from app.services.skills import SKILLS
from app.services.provider_service import provider_configuration
from app.ui.handlers import status_markdown, refresh_provider, rag_answer, ingest_file
from app.ui.styles import CSS, EMPTY_CHAT, INITIAL_THEME, TOGGLE_THEME, DARK_HEAD, build_theme


def create_demo():
    default = provider_configuration("pinecone")
    limits = AppSettings()
    with gr.Blocks(title="Generative AI-Powered Retrieval System for Technical Documentation⁠", analytics_enabled=False, delete_cache=(3600, 86400)) as demo:
        receipts = gr.State({"shared": []})
        with gr.Sidebar(label="Workspace", open=True, width=320, elem_id="workspace-sidebar") as sidebar:
            gr.Markdown("## Knowledge AI")
            with gr.Column(elem_id="session-list") as history_slot:
                gr.Markdown("### Chats")
            gr.Markdown("### Settings")
            dark_mode = gr.Checkbox(value=True, label="Dark mode")
            dark_mode.change(fn=None, inputs=dark_mode, outputs=None, js=TOGGLE_THEME, queue=False)
            provider = gr.State("pinecone")
            gr.Markdown("**Pinecone Assistant**")
            connection_status = gr.Markdown(status_markdown(default), elem_id="provider-status")
            skill = gr.Dropdown(choices=[(policy.label, name) for name, policy in SKILLS.items()],
                                value="general", label="Pinecone task", info="Applies to the next Pinecone answer.")
            web_search = gr.Checkbox(value=False, label="Allow web search", info="Pinecone only: search when documents cannot answer.")
            gr.Markdown("### Document library")
            library_note = gr.Markdown("Files upload to Pinecone Assistant. Reload the page to update processing status.")
            upload = gr.File(label=f"PDF or TXT · up to {limits.MAX_UPLOAD_MB} MB", elem_id="document-upload", file_types=[".pdf", ".txt"], type="filepath")
            upload_button = gr.Button("Upload to Pinecone Assistant", variant="primary")
            upload_status = gr.Markdown("Files upload only to Pinecone Assistant. Wait until their status is Available before chatting.")
            library = gr.Dataframe(headers=["Document", "Status"], datatype=["str", "str"], value=[], interactive=False, label="Your documents", elem_id="document-library")
        with gr.Column(scale=3, min_width=320, elem_id="conversation-panel"):
            gr.Markdown("### Ask your documents\nClear answers, grounded in your knowledge.")
            chatbot = gr.Chatbot(label="Document answers", height="65vh", placeholder=EMPTY_CHAT, elem_id="knowledge-chat")
            response_status = gr.Markdown("Pinecone Assistant - Ready" if default["configured"] else "Pinecone Assistant - Setup required", elem_id="response-status")
            chat_interface = gr.ChatInterface(
                fn=rag_answer, chatbot=chatbot, save_history=True,
                additional_inputs=[provider, web_search, skill],
                additional_outputs=[response_status],
                # Message-only examples retain the current provider and controls.
                examples=[["Summarize the key points"], ["Explain a technical concept"],
                          ["Find a specific requirement"]],
                run_examples_on_click=True, cache_examples=False,
                submit_btn="Send", stop_btn="Stop", analytics_enabled=False,
                concurrency_limit=limits.UI_CONCURRENCY,
            )
            gr.Markdown("Pinecone Assistant answers from its uploaded files. Each question is independent.")
            gr.HTML('<a href="/docs" target="_blank">API documentation</a> · <a href="/health" target="_blank">System health</a>')
        # Reuse Gradio's native history events and browser storage; move its history
        # controls into our unified sidebar instead of maintaining a second chat store.
        history_area = chat_interface.chat_history_dataset.parent
        history_area.unrender()
        with sidebar:
            with history_slot:
                history_area.render()
                gr.Markdown("Saved in this browser. Questions are answered independently.")
        chat_interface.new_chat_button.value = "New chat"
        chat_interface.chat_history_dataset.elem_id = "saved-chats"
        # Event wiring coordinates views; Gradio supplies event dispatch.
        chat_interface.textbox.stop(
            lambda: "Response stopped. Partial answer kept.", outputs=response_status, queue=False,
        )
        chatbot.clear(lambda: "Ready for a new question.", outputs=response_status, queue=False)
        # Serialize library updates while an upload or refresh is running.
        def lock_controls():
            return gr.update(interactive=False)
        def unlock_controls():
            return gr.update(interactive=True)
        demo.load(lock_controls, outputs=upload_button, queue=False).then(
            refresh_provider, inputs=[provider, receipts], outputs=[connection_status, receipts, library],
        ).then(unlock_controls, outputs=upload_button, queue=False)
        upload_button.click(lock_controls, outputs=upload_button, queue=False).then(
            ingest_file, inputs=[upload, receipts, provider], outputs=[upload_status, receipts, library],
        ).then(
            refresh_provider, inputs=[provider, receipts], outputs=[connection_status, receipts, library],
        ).then(unlock_controls, outputs=upload_button, queue=False)
    demo.queue(max_size=limits.UI_QUEUE_SIZE, default_concurrency_limit=limits.UI_CONCURRENCY)
    return demo


def mount_demo(app, demo):
    return gr.mount_gradio_app(
        app, demo, path="/", css=CSS, js=INITIAL_THEME, head=DARK_HEAD,
        max_file_size=AppSettings().max_upload_bytes, show_error=False, footer_links=[],
        theme=build_theme(),
    )
