import type { Metadata } from "next";
import localFont from "next/font/local";
import "./globals.css";
import { AuthProvider } from "@/lib/auth-context";
import { AppShell } from "@/components/app-shell";

// Fonts ship in the repo (latin subset, OFL licenses alongside) instead of
// next/font/google, which downloads them during `next build`. A changed Google
// response broke the Turbopack build in CI with no code change on our side.
const dmSans = localFont({
  src: "./fonts/dm-sans-latin.woff2",
  variable: "--font-dm-sans",
  weight: "400 700",
});

const jetbrains = localFont({
  src: "./fonts/jetbrains-mono-latin.woff2",
  variable: "--font-jetbrains",
  weight: "400 700",
});

const barlow = localFont({
  src: [
    { path: "./fonts/barlow-condensed-500-latin.woff2", weight: "500" },
    { path: "./fonts/barlow-condensed-600-latin.woff2", weight: "600" },
    { path: "./fonts/barlow-condensed-700-latin.woff2", weight: "700" },
    { path: "./fonts/barlow-condensed-800-latin.woff2", weight: "800" },
  ],
  variable: "--font-display",
});

export const metadata: Metadata = {
  title: "NFL Algorithm | Pro Betting Dashboard",
  description:
    "Professional NFL value betting dashboard with ML-powered predictions",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body
        className={`${dmSans.variable} ${jetbrains.variable} ${barlow.variable} antialiased bg-[#0a0e17] text-slate-100 font-[family-name:var(--font-dm-sans)]`}
      >
        <AuthProvider>
          <AppShell>{children}</AppShell>
        </AuthProvider>
      </body>
    </html>
  );
}
