// components/Chat.tsx
"use client";

import { useState, useEffect, useRef } from "react";
import Upload from "@/components/Upload";

interface ChatSession {
  id: string;
  title: string;
}

export default function Chat() {
  const [inputMessage, setInputMessage] = useState<string>("");
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [backendResponse, setBackendResponse] = useState<string>("");

  // Saved Conversations History (Backend DB persistence)
  const [chatSessions, setChatSessions] = useState<ChatSession[]>([]);
  const [activeChatId, setActiveChatId] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState<boolean>(true);
  
  const chatBottomRef = useRef<HTMLDivElement>(null);

  // Fetch saved conversations from backend on initial mount
  useEffect(() => {
    fetchChatHistory();
  }, []);

  // Auto scroll on response update
  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [backendResponse, isLoading]);

  const fetchChatHistory = async () => {
    try {
      const res = await fetch("http://127.0.0.1:8000/chat/history");
      if (res.ok) {
        const data = await res.json();
        setChatSessions(data.sessions || []);
      }
    } catch {
      // Fallback local memory list if database endpoint isn't mounted yet
      console.log("Database history endpoint unavailable, using local session memory.");
    }
  };

  const handleNewChat = async () => {
    const newSessionId = Date.now().toString();
    const newSession: ChatSession = { id: newSessionId, title: "New Conversation" };

    setActiveChatId(newSessionId);
    setInputMessage("");
    setBackendResponse("");

    // Persist new chat creation to database
    try {
      await fetch("http://127.0.0.1:8000/chat/session", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(newSession),
      });
      fetchChatHistory();
    } catch {
      setChatSessions((prev) => [newSession, ...prev]);
    }
  };

  const sendMessage = async () => {
    if (!inputMessage.trim()) return;

    setIsLoading(true);
    setBackendResponse(""); // Clear previous data before fetching new data

    try {
      const res = await fetch("http://127.0.0.1:8000/chat/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ 
          query: inputMessage,
          chat_id: activeChatId || "default"
        }),
      });

      if (!res.ok) {
        throw new Error(`Server error: ${res.status}`);
      }

      const data = await res.json();
      console.log(data);
      
      // Update state with incoming data
      setBackendResponse(data.answer || JSON.stringify(data.answer));

      // Refresh chat sidebar titles from DB after first turn
      fetchChatHistory();

    } catch (error) {
      setBackendResponse(`Failed to fetch data: ${error instanceof Error ? error.message : "Unknown error"}`);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex h-screen bg-[#212121] text-gray-100 font-sans antialiased overflow-hidden">
      {/* Sidebar - Authentic ChatGPT Left Dock */}
      <aside className={`${sidebarOpen ? "w-64" : "w-0 overflow-hidden"} transition-all duration-300 flex flex-col bg-[#171717] border-r border-white/10 select-none`}>
        <div className="p-3 space-y-3">
          <button 
            onClick={handleNewChat}
            className="flex items-center justify-between w-full p-2.5 rounded-lg border border-white/20 hover:bg-white/5 transition-colors text-sm font-medium text-white"
          >
            <span className="flex items-center gap-2">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 4v16m8-8H4" />
              </svg>
              New chat
            </span>
            <svg className="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
            </svg>
          </button>

          {/* RAG File Upload Drawer */}
          <div className="pt-2">
            <Upload />
          </div>
        </div>

        {/* Database Stored Chat History List */}
        <div className="flex-1 overflow-y-auto space-y-1 px-3">
          <div className="px-3 py-2 text-xs font-semibold text-gray-400 uppercase tracking-wider">Recent Chats</div>
          {chatSessions.length === 0 ? (
            <div className="px-3 py-2 text-xs text-gray-500 italic">No saved conversations</div>
          ) : (
            chatSessions.map((session) => (
              <div
                key={session.id}
                onClick={() => setActiveChatId(session.id)}
                className={`px-3 py-2.5 text-sm rounded-lg truncate cursor-pointer transition-colors ${
                  activeChatId === session.id ? "bg-white/10 text-white font-medium" : "text-gray-300 hover:bg-white/5"
                }`}
              >
                {session.title}
              </div>
            ))
          )}
        </div>

        {/* Profile Footer */}
        <div className="p-3 border-t border-white/10 flex items-center gap-3">
          <div className="w-8 h-8 rounded-full bg-purple-600 flex items-center justify-center font-bold text-xs text-white">
            U
          </div>
          <div className="text-sm font-medium truncate text-gray-200">Test User</div>
        </div>
      </aside>

      {/* Main Workspace */}
      <main className="flex-1 flex flex-col h-full bg-[#212121] relative overflow-hidden">
        {/* Top Header Navigation */}
        <header className="h-14 border-b border-white/10 flex items-center justify-between px-4 bg-[#212121]">
          <div className="flex items-center gap-3">
            <button 
              onClick={() => setSidebarOpen(!sidebarOpen)} 
              className="p-1.5 rounded-lg text-gray-400 hover:text-white hover:bg-white/10 transition-colors"
              title="Toggle Sidebar"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>

            <div className="flex items-center gap-2 cursor-pointer hover:bg-white/5 px-2 py-1 rounded-lg transition-colors">
              <h1 className="text-base font-semibold text-gray-200">ChatGPT</h1>
              <span className="text-xs bg-gray-800 text-gray-400 px-2 py-0.5 rounded border border-gray-700 font-mono">
                RAG Test Mode
              </span>
            </div>
          </div>
        </header>

        {/* Scrollable Conversation Container */}
        <div className="flex-1 overflow-y-auto p-4 md:p-6 space-y-6 max-w-3xl w-full mx-auto">
          {/* Default Hero State */}
          {!inputMessage && !backendResponse && !isLoading && (
            <div className="h-full flex flex-col items-center justify-center text-center my-auto min-h-[50vh] text-gray-400">
              <div className="w-12 h-12 rounded-full bg-white/5 flex items-center justify-center mb-4 border border-white/10">
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                </svg>
              </div>
              <h2 className="text-2xl font-semibold text-white mb-2">What can I help with today?</h2>
              <p className="text-sm text-gray-400 max-w-sm">
                Type your prompt below to send a query directly to your RAG backend connection.
              </p>
            </div>
          )}

          {/* User Message Bubble */}
          {inputMessage && (
            <div className="flex justify-end gap-4 max-w-3xl mx-auto">
              <div className="bg-[#303030] text-gray-100 px-4 py-3 rounded-2xl max-w-[85%] text-sm leading-relaxed shadow-sm">
                {inputMessage}
              </div>
            </div>
          )}

          {/* Loading Indicator */}
          {isLoading && (
            <div className="flex gap-4 max-w-3xl mx-auto">
              <div className="w-8 h-8 rounded-full bg-emerald-600 flex-shrink-0 flex items-center justify-center text-white text-xs font-bold">
                AI
              </div>
              <div className="flex items-center space-x-2 bg-transparent py-2">
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:-0.3s]"></div>
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:-0.15s]"></div>
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
              </div>
            </div>
          )}

          {/* Assistant Response Container */}
          {backendResponse && (
            <div className="flex gap-4 max-w-3xl mx-auto">
              <div className="w-8 h-8 rounded-full bg-emerald-600 flex-shrink-0 flex items-center justify-center text-white text-xs font-bold shadow">
                AI
              </div>
              <div className="flex-1 space-y-2">
                <span className="text-xs font-semibold text-gray-400 uppercase tracking-wide">
                  Backend Response
                </span>
                <div className="text-gray-200 text-sm leading-relaxed bg-[#2f2f2f]/60 p-4 rounded-xl border border-white/5 whitespace-pre-wrap font-sans">
                  {backendResponse}
                </div>
              </div>
            </div>
          )}

          <div ref={chatBottomRef} />
        </div>

        {/* Input Bar Dock Area */}
        <div className="p-4 bg-[#212121]">
          <div className="max-w-3xl mx-auto relative">
            <div className="relative flex items-center bg-[#2f2f2f] rounded-3xl border border-white/10 shadow-lg focus-within:border-white/20 transition-all">
              <input
                type="text"
                className="w-full bg-transparent text-gray-100 placeholder-gray-400 text-sm px-5 py-3.5 pr-12 focus:outline-none"
                placeholder="Type a test query..."
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !isLoading) {
                    sendMessage();
                  }
                }}
              />

              <button
                onClick={sendMessage}
                disabled={isLoading || !inputMessage.trim()}
                className="absolute right-2.5 p-2 rounded-full bg-white text-black hover:bg-gray-200 disabled:bg-gray-600 disabled:text-gray-400 disabled:cursor-not-allowed transition-all"
                title="Send query"
              >
                {isLoading ? (
                  <svg className="w-4 h-4 animate-spin" viewBox="0 0 24 24" fill="none">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path>
                  </svg>
                ) : (
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M5 10l7-7m0 0l7 7m-7-7v18" />
                  </svg>
                )}
              </button>
            </div>
            
            <p className="text-[11px] text-center text-gray-500 mt-2">
              ChatGPT-style interface for RAG connectivity tests.
            </p>
          </div>
        </div>
      </main>
    </div>
  );
}