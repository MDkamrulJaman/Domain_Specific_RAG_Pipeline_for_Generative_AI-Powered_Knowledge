"""Consistent document parsing from files and in-memory uploads."""




def test_file_loader_uses_same_parser_for_disk_and_bytes(tmp_path):
    from app.pipeline.loader import UniversalDocumentLoader
    path = tmp_path / "a.txt"
    path.write_text("Document text", encoding="utf-8")
    loader = UniversalDocumentLoader()
    assert loader.load_file(path)[0].page_content == loader.load_bytes("a.txt", b"Document text")[0].page_content
