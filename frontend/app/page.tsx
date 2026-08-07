// app/page.tsx
"use client";

import Chat from "@/components/Chat";

export default function Home() {
  return (
    <div className="h-screen w-screen overflow-hidden bg-[#212121]">
      <Chat />
    </div>
  );
}