"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { cn } from "@/lib/utils";
import {
  LayoutDashboard,
  TrendingUp,
  BarChart3,
  Settings,
  Activity,
  ChevronLeft,
  ChevronRight,
  LogOut,
  User,
} from "lucide-react";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { useAuth } from "@/lib/auth-context";
import { PerformanceWidget } from "@/components/performance-widget";

interface NavItem {
  title: string;
  href: string;
  icon: React.ComponentType<{ className?: string }>;
}

const navItems: NavItem[] = [
  { title: "Dashboard", href: "/", icon: LayoutDashboard },
  { title: "Performance", href: "/performance", icon: TrendingUp },
  { title: "Analytics", href: "/analytics", icon: BarChart3 },
  { title: "System", href: "/system", icon: Activity },
];

export function Sidebar() {
  const pathname = usePathname();
  const router = useRouter();
  const { user, logout } = useAuth();
  const [collapsed, setCollapsed] = useState(false);

  const handleLogout = async () => {
    await logout();
    router.push("/login");
  };

  return (
    <aside
      className={cn(
        "flex flex-col h-screen bg-[#0d1220] border-r border-slate-800/50 transition-all duration-300",
        collapsed ? "w-16" : "w-64"
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between h-16 px-4 border-b border-slate-800/50">
        {!collapsed && (
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-md bg-gradient-to-br from-amber-500 to-amber-700 flex items-center justify-center">
              <span className="text-xs font-bold text-black font-[family-name:var(--font-jetbrains)]">
                NFL
              </span>
            </div>
            <div className="flex flex-col">
              <span className="text-sm font-semibold text-slate-100 tracking-tight">
                NFL Algorithm
              </span>
              <span className="text-[10px] text-amber-500/70 font-[family-name:var(--font-jetbrains)] uppercase tracking-widest">
                v2.1 Pro
              </span>
            </div>
          </div>
        )}
        {collapsed && (
          <div className="w-8 h-8 rounded-md bg-gradient-to-br from-amber-500 to-amber-700 flex items-center justify-center mx-auto">
            <span className="text-xs font-bold text-black font-[family-name:var(--font-jetbrains)]">
              N
            </span>
          </div>
        )}
        <Button
          variant="ghost"
          size="icon"
          onClick={() => setCollapsed(!collapsed)}
          className={cn(
            "h-7 w-7 text-slate-500 hover:text-slate-300 hover:bg-slate-800/50",
            collapsed && "hidden"
          )}
        >
          <ChevronLeft className="h-3.5 w-3.5" />
        </Button>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-2 py-4 space-y-0.5">
        {!collapsed && (
          <p className="px-3 mb-3 text-[10px] font-medium text-slate-600 uppercase tracking-[0.15em]">
            Navigation
          </p>
        )}
        {navItems.map((item) => {
          const isActive = pathname === item.href;
          const Icon = item.icon;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium transition-all duration-150",
                isActive
                  ? "bg-amber-500/10 text-amber-400 border-l-2 border-amber-500 ml-0"
                  : "text-slate-500 hover:bg-slate-800/40 hover:text-slate-300"
              )}
            >
              <Icon
                className={cn(
                  "h-4 w-4 flex-shrink-0",
                  isActive ? "text-amber-400" : ""
                )}
              />
              {!collapsed && <span>{item.title}</span>}
            </Link>
          );
        })}
      </nav>

      {/* Performance Widget */}
      <PerformanceWidget collapsed={collapsed} />

      <Separator className="bg-slate-800/50" />

      {/* Footer */}
      <div className="p-3 space-y-1">
        <Link
          href="/settings"
          className={cn(
            "flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium text-slate-500 hover:bg-slate-800/40 hover:text-slate-300 transition-colors",
            pathname === "/settings" &&
              "bg-amber-500/10 text-amber-400"
          )}
        >
          <Settings className="h-4 w-4 flex-shrink-0" />
          {!collapsed && <span>Settings</span>}
        </Link>

        {user ? (
          <>
            <div className="flex items-center gap-3 px-3 py-2">
              <div className="h-7 w-7 rounded-full bg-gradient-to-br from-slate-600 to-slate-700 flex items-center justify-center flex-shrink-0 ring-1 ring-slate-700">
                <User className="h-3.5 w-3.5 text-slate-300" />
              </div>
              {!collapsed && (
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-slate-200 truncate">
                    {user.name || user.email.split("@")[0]}
                  </p>
                  <p className="text-[11px] text-slate-600 truncate">
                    {user.email}
                  </p>
                </div>
              )}
            </div>
            <button
              onClick={handleLogout}
              className="flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium text-slate-500 hover:bg-red-500/10 hover:text-red-400 transition-colors w-full"
            >
              <LogOut className="h-4 w-4 flex-shrink-0" />
              {!collapsed && <span>Logout</span>}
            </button>
          </>
        ) : (
          <Link
            href="/login"
            className="flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium text-slate-500 hover:bg-slate-800/40 hover:text-slate-300 transition-colors"
          >
            <User className="h-4 w-4 flex-shrink-0" />
            {!collapsed && <span>Sign In</span>}
          </Link>
        )}
      </div>
    </aside>
  );
}
