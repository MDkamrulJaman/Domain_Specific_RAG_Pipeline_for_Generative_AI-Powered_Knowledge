"""UI provider selection, streaming callbacks, library state, and dark theme."""

from unittest.mock import Mock
import asyncio


def test_gradio_selector_and_streaming_callback(monkeypatch):
    from app.ui import handlers as frontend
    from app.ui.frontend import create_demo
    demo = create_demo()
    dropdowns = [block for block in demo.blocks.values() if getattr(block, "label", None) == "Answer provider"]
    assert len(dropdowns) == 1
    assert dropdowns[0].value == "pinecone"
    service = Mock()
    service.stream.return_value = iter(["first", " second"])
    factory = Mock(return_value=service)
    monkeypatch.setattr(frontend, "get_chat_service", factory)
    updates = list(frontend.rag_answer("q", [], "pinecone"))
    assert [item[0] for item in updates] == ["", "first", "first second", "first second"]
    assert "Response complete" in updates[-1][1]
    factory.assert_called_once_with("pinecone")


def test_frontend_switch_selects_only_provider_library(monkeypatch):
    from app.ui import handlers as frontend
    monkeypatch.setattr(frontend,"provider_configuration",lambda p:{"provider":p,"label":p,"configured":True,"model":"m","message":"Configured","enable_thinking":False})
    receipts={"shared":[{"name":"n.txt","results":{"nvidia":{"status":"Indexed"}}},{"name":"p.txt","results":{"pinecone":{"status":"Processing"}}}]}
    p=frontend.provider_panel("pinecone",receipts)
    n=frontend.provider_panel("nvidia",receipts)
    assert p[4] == [["p.txt","Processing"]]
    assert n[4] == [["n.txt","Indexed"]]
    assert not p[1]["visible"] and n[1]["visible"]
    assert p[3]["value"] == "Upload to Pinecone Assistant"
    assert n[3]["value"] == "Upload to NVIDIA model"


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


def test_refresh_removes_deleted_assistant_files_but_preserves_nvidia(monkeypatch):
    from app.ui import handlers as frontend
    monkeypatch.setattr(frontend,"inspect_provider",lambda _: {"provider":"pinecone","label":"Assistant","configured":True,"connected":True,"model":"m","message":"Ready","files":[{"file_id":"new","name":"new.txt","status":"Available"}]})
    receipts={"shared":[{"name":"old.txt","results":{"pinecone":{"file_id":"old","status":"Available"},"nvidia":{"status":"Indexed"}}}]}
    _,updated,rows=frontend.refresh_provider("pinecone",receipts)
    assert rows == [["new.txt","Available"]]
    assert frontend.library_rows("nvidia",updated) == [["old.txt","Indexed"]]
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
    assert interface.additional_inputs[0].label == "Answer provider"
    assert interface.additional_inputs[0].value == "pinecone"


def test_provider_change_automatically_refreshes_without_button():
    from app.ui.frontend import create_demo
    from app.ui.handlers import refresh_provider
    demo = create_demo()
    assert not any(getattr(block, "value", None) == "Refresh connection & files"
                   for block in demo.blocks.values() if isinstance(getattr(block, "value", None), str))
    provider = next(b for b in demo.blocks.values() if getattr(b, "label", None) == "Answer provider")
    change = next(fn for fn in demo.fns.values() if (provider._id, "change") in fn.targets)
    panel = next(fn for fn in demo.fns.values() if fn.trigger_after == change._id)
    refresh = next(fn for fn in demo.fns.values() if fn.trigger_after == panel._id)
    assert refresh.fn is refresh_provider
    assert refresh.inputs[0] is provider
    assert any(getattr(output, "elem_id", None) == "document-library" for output in refresh.outputs)
