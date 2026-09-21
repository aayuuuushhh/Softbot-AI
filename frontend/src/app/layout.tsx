import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "UDDHAR — Satellite Disaster-Damage Triage",
  // No favicon: an empty data URI stops the browser falling back to a
  // /favicon.ico request that this app does not serve.
  icons: { icon: "data:," },
  description:
    "AI-powered building damage assessment from pre/post disaster satellite imagery. Deterministic ML scoring, ranked zone triage, English situation briefs. Team DarkNem — AMD Hackathon ACT II.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-diq-bg text-diq-ink antialiased">
        {children}
      </body>
    </html>
  );
}
