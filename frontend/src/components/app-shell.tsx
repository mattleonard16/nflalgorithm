"use client";

import { usePathname } from "next/navigation";
import Link from "next/link";
import { isNavActive, nflNavItems, Sidebar } from "@/components/sidebar";
import { useAuth } from "@/lib/auth-context";
import { cn } from "@/lib/utils";

const publicRoutes = ["/login", "/signup"];

const mobileHrefs = new Set(["/", "/analytics", "/bets", "/system"]);
const mobileItems = nflNavItems
  .filter((item) => mobileHrefs.has(item.href))
  .map((item) => (item.href === "/" ? { ...item, title: "Board" } : item));

function MobileDock({ pathname }: { pathname: string }) {
  return (
    <nav aria-label="Mobile navigation" className="fixed inset-x-3 bottom-3 z-50 grid grid-cols-4 rounded-2xl border border-slate-700/60 bg-[#0d1220]/95 p-1.5 shadow-2xl shadow-black/40 backdrop-blur-xl md:hidden">
      {mobileItems.map((item) => {
        const Icon = item.icon;
        const active = isNavActive(pathname, item.href);
        return (
          <Link
            key={item.href}
            href={item.href}
            className={cn(
              "flex min-h-12 flex-col items-center justify-center gap-1 rounded-xl text-[9px] font-semibold uppercase tracking-wider transition-colors",
              active ? "bg-amber-400/10 text-amber-300" : "text-slate-500 hover:bg-slate-800/60 hover:text-slate-300"
            )}
          >
            <Icon className="h-4 w-4" />
            {item.title}
          </Link>
        );
      })}
    </nav>
  );
}

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const { loading } = useAuth();

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen bg-[#0a0e17]">
        <div className="flex flex-col items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-amber-500 to-amber-700 flex items-center justify-center animate-pulse">
            <span className="text-xs font-bold text-black font-[family-name:var(--font-jetbrains)]">
              NFL
            </span>
          </div>
          <span className="text-sm text-slate-500">Loading...</span>
        </div>
      </div>
    );
  }

  if (publicRoutes.includes(pathname)) {
    return <>{children}</>;
  }

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      <main className="flex-1 overflow-auto bg-grid">
        <div className="mx-auto max-w-[1400px] px-4 py-4 pb-24 sm:px-6 sm:py-6 md:pb-6">{children}</div>
      </main>
      <MobileDock pathname={pathname} />
    </div>
  );
}
