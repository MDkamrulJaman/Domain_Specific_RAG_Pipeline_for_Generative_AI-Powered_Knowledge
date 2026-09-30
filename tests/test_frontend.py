"""UI provider selection, streaming callbacks, library state, and dark theme."""

from unittest.mock import Mock
import asyncio


def test_gradio_selector_and_streaming_callback(monkeypatch):
    from app.ui import handlers as frontend
    from app.ui.frontend import create_demo
    demo = create_demo()
    dropdowns = [block for block in demo.blocks.values() if getattr(block, "label", None) == "Answer provider"]
    assert dropdowns == []
    service = Mock()
    service.stream.return_value = iter(["first", " second"])
    factory = Mock(return_value=service)
    monkeypatch.setattr(frontend, "get_chat_service", factory)
    updates = list(frontend.rag_answer("q", [], "pinecone"))
    assert [item[0] for item in updates] == ["", "first", "first second", "first second"]
    assert "Response complete" in updates[-1][1]
    factory.assert_called_once_with("pinecone")


def test_frontend_upload_tracks_shared_library(tmp_path, monkeypatch):
    import asyncio
    from app.ui import handlers as frontend
    path = tmp_path / "manual.txt"
    path.write_text("sample content")
    calls = []
    def process(name, content, target, *, progress):
        calls.append((name, content))
        progress(0.5, "Processing")
        return {"status": "success", "message": "Processing", "results": {target: {"status":"Processing"}}}
    monkeypatch.setattr(frontend, "process_document", process)
    original = {"shared": []}
    message, receipts, rows = asyncio.run(frontend.ingest_file(str(path), original, progress=lambda *a, **kw: None))
    assert calls == [("manual.txt", b"sample content")]
    assert "Upload results" in message
    assert original == {"shared": []}
    assert len(receipts["shared"]) == 1
    assert frontend.library_rows("nvidia", receipts) == []
    assert frontend.library_rows("pinecone", receipts) == [["manual.txt","Processing"]]


def test_refresh_removes_deleted_assistant_files(monkeypatch):
    from app.ui import handlers as frontend
    monkeypatch.setattr(frontend,"inspect_provider",lambda _: {"provider":"pinecone","label":"Assistant","configured":True,"connected":True,"model":"m","message":"Ready","files":[{"file_id":"new","name":"new.txt","status":"Available"}]})
    receipts={"shared":[{"name":"old.txt","results":{"pinecone":{"file_id":"old","status":"Available"}}}]}
    _,updated,rows=frontend.refresh_provider("pinecone",receipts)
    assert rows == [["new.txt","Available"]]
    assert len(updated["shared"]) == 1
    assert "pinecone" in receipts["shared"][0]["results"]


def test_theme_and_mount_apply_dark_mode_and_upload_limit(monkeypatch):
    from app.ui import frontend, styles
    assert styles.build_theme() is not None
    mount = Mock()
    monkeypatch.setattr(frontend.gr, "mount_gradio_app", mount)
    frontend.mount_demo(Mock(), Mock())
    options = mount.call_args.kwargs
    assert options["js"] == styles.INITIAL_THEME
    assert "classList.add('dark')" in options["head"]
    assert options["max_file_size"] == 5 * 1024 * 1024
    assert options["show_error"] is False


def test_suggestions_submit_using_live_provider_controls():
    from app.ui.frontend import create_demo
    from app.ui.handlers import rag_answer
    demo = create_demo()
    chatbot = next(block for block in demo.blocks.values() if getattr(block, "elem_id", None) == "knowledge-chat")
    assert [example["text"] for example in chatbot.examples] == [
        "Summarize the key points", "Explain a technical concept", "Find a specific requirement",
    ]
    selection = next(fn for fn in demo.fns.values() if (chatbot._id, "example_select") in fn.targets)
    interface = selection.fn.__self__
    assert interface.run_examples_on_click is True
    assert interface.cache_examples is False
    assert interface.fn is rag_answer
    assert interface._additional_inputs_in_examples is False
    assert interface.additional_inputs[0].value == "pinecone"


def test_sidebar_contains_history_controls_and_documents():
    from app.ui.frontend import create_demo
    import gradio as gr
    demo = create_demo()
    sidebar = next(b for b in demo.blocks.values() if isinstance(b, gr.Sidebar))
    for label in ("Pinecone task", "Allow web search", "Your documents"):
        component = next(b for b in demo.blocks.values() if getattr(b,"label",None) == label)
        parent = component.parent
        while parent is not None and parent is not sidebar:
            parent = parent.parent
        assert parent is sidebar
    history = next(b for b in demo.blocks.values() if getattr(b,"elem_id",None) == "saved-chats")
    assert history.parent.parent.parent is sidebar
    assert any(getattr(b,"value",None) == "New chat" for b in demo.blocks.values() if isinstance(getattr(b,"value",None), str))


def test_native_sessions_save_load_new_and_delete():
    from app.ui.frontend import create_demo
    demo = create_demo()
    chatbot = next(b for b in demo.blocks.values() if getattr(b,"elem_id",None) == "knowledge-chat")
    selection = next(fn for fn in demo.fns.values() if (chatbot._id,"example_select") in fn.targets)
    interface = selection.fn.__self__
    assert interface.save_history is True
    first = [{"role":"user","content":"First question"},{"role":"assistant","content":"Answer"}]
    second = [{"role":"user","content":"Second question"}]
    index, saved = interface._save_conversation(None, first, [])
    assert index == 0 and saved[0] == first
    index, saved = interface._save_conversation(None, second, saved)
    assert len(saved) == 2
    restored = interface._load_conversation(1, saved)[1].value
    assert [message["role"] for message in restored] == ["user", "assistant"]
    assert [message["content"][0]["text"] for message in restored] == ["First question", "Answer"]
    _, saved = interface._delete_conversation(0, saved)
    assert saved == [first]
