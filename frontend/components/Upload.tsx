"use client";

import { useState, useRef } from "react";

interface UploadProps {
  onUploadSuccess?: () => void;
}

export default function Upload({ onUploadSuccess }: UploadProps) {
  const [file, setFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleUpload = async () => {
    if (!file) return;

    setIsUploading(true);
    setUploadStatus(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://127.0.0.1:8000/ingest/upload", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error(`Upload failed with status ${res.status}`);

      setUploadStatus("Success");
      setFile(null);
      if (fileInputRef.current) fileInputRef.current.value = "";
      if (onUploadSuccess) onUploadSuccess();
    } catch (error) {
      setUploadStatus(error instanceof Error ? error.message : "Error uploading file");
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="bg-[#2f2f2f] p-3 rounded-xl border border-white/10 text-xs text-gray-200">
      <div className="flex flex-col gap-2">
        <label className="font-semibold text-gray-300">Ingest RAG Knowledge</label>
        
        <input
          ref={fileInputRef}
          type="file"
          onChange={(e) => {
            setFile(e.target.files?.[0] || null);
            setUploadStatus(null);
          }}
          className="text-xs text-gray-400 file:mr-2 file:py-1 file:px-2 file:rounded-md file:border-0 file:text-xs file:font-semibold file:bg-white/10 file:text-gray-200 hover:file:bg-white/20 cursor-pointer"
        />

        {file && (
          <button
            onClick={handleUpload}
            disabled={isUploading}
            className="w-full py-1.5 px-3 rounded-lg bg-emerald-600 hover:bg-emerald-500 disabled:bg-gray-600 font-medium text-white transition-colors flex items-center justify-center gap-2"
          >
            {isUploading ? (
              <>
                <div className="w-3 h-3 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                Uploading...
              </>
            ) : (
              `Upload ${file.name}`
            )}
          </button>
        )}

        {uploadStatus && (
          <span className={`text-[11px] ${uploadStatus === "Success" ? "text-emerald-400" : "text-rose-400"}`}>
            {uploadStatus === "Success" ? "✓ Document successfully ingested!" : uploadStatus}
          </span>
        )}
      </div>
    </div>
  );
}