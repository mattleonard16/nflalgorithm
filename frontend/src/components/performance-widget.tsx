"use client";

import { useEffect, useState } from "react";
import { TrendingUp, TrendingDown, ChevronDown, ChevronUp } from "lucide-react";
import { cn } from "@/lib/utils";
import { getWeeklySummary } from "@/lib/api";
import type { WeeklySummaryItem } from "@/lib/types";

interface PerformanceWidgetProps {
    collapsed?: boolean;
}

export function PerformanceWidget({ collapsed = false }: PerformanceWidgetProps) {
    const [weeks, setWeeks] = useState<WeeklySummaryItem[]>([]);
    const [loading, setLoading] = useState(true);
    const [expanded, setExpanded] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        async function fetchData() {
            try {
                const data = await getWeeklySummary(4);
                setWeeks(data.weeks);
                setError(null);
            } catch {
                setError("Failed to load");
            } finally {
                setLoading(false);
            }
        }
        fetchData();
    }, []);

    if (collapsed) {
        return null;
    }

    if (loading) {
        return (
            <div className="px-4 py-3">
                <div className="animate-pulse space-y-2">
                    <div className="h-4 bg-zinc-800 rounded w-3/4"></div>
                    <div className="h-3 bg-zinc-800 rounded w-1/2"></div>
                </div>
            </div>
        );
    }

    if (error || weeks.length === 0) {
        return null;
    }

    return (
        <div className="px-2 py-3">
            {/* Header */}
            <button
                onClick={() => setExpanded(!expanded)}
                className="flex items-center justify-between w-full px-3 py-2 text-xs font-medium text-zinc-500 uppercase tracking-wider hover:text-zinc-400 transition-colors"
            >
                <span>Recent Performance</span>
                {expanded ? (
                    <ChevronUp className="h-3 w-3" />
                ) : (
                    <ChevronDown className="h-3 w-3" />
                )}
            </button>

            {expanded && (
                <div className="space-y-2 mt-1">
                    {weeks.map((week) => (
                        <WeekCard key={`${week.season}-${week.week}`} week={week} />
                    ))}

                    {/* Edge Tier Summary */}
                    <EdgeTierSummary weeks={weeks} />
                </div>
            )}
        </div>
    );
}

function WeekCard({ week }: { week: WeeklySummaryItem }) {
    const hasResults = week.wins + week.losses > 0;
    const isPositive = week.roi_pct > 0;

    return (
        <div className="px-3 py-2 rounded-lg bg-zinc-900/50 border border-zinc-800">
            <div className="flex items-center justify-between mb-1">
                <span className="text-sm font-medium text-zinc-200">
                    Week {week.week}
                </span>
                {hasResults ? (
                    <span
                        className={cn(
                            "text-xs font-semibold flex items-center gap-1",
                            isPositive ? "text-emerald-400" : "text-red-400"
                        )}
                    >
                        {isPositive ? (
                            <TrendingUp className="h-3 w-3" />
                        ) : (
                            <TrendingDown className="h-3 w-3" />
                        )}
                        {isPositive ? "+" : ""}
                        {week.roi_pct.toFixed(1)}%
                    </span>
                ) : (
                    <span className="text-xs text-zinc-500">Pending</span>
                )}
            </div>

            {hasResults ? (
                <>
                    <div className="flex items-center justify-between text-xs text-zinc-400 mb-1">
                        <span>
                            {week.wins}-{week.losses}
                            {week.pushes > 0 && `-${week.pushes}`}
                        </span>
                        <span>{week.win_rate.toFixed(0)}% win</span>
                    </div>
                    {/* Progress bar */}
                    <div className="h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                        <div
                            className={cn(
                                "h-full rounded-full transition-all",
                                week.win_rate >= 60
                                    ? "bg-emerald-500"
                                    : week.win_rate >= 50
                                        ? "bg-yellow-500"
                                        : "bg-red-500"
                            )}
                            style={{ width: `${Math.min(week.win_rate, 100)}%` }}
                        />
                    </div>
                </>
            ) : (
                <div className="text-xs text-zinc-500">
                    {week.total_picks} picks awaiting results
                </div>
            )}
        </div>
    );
}

function EdgeTierSummary({ weeks }: { weeks: WeeklySummaryItem[] }) {
    // Aggregate edge tier performance across all weeks
    const tierStats = new Map<string, { wins: number; losses: number }>();

    for (const week of weeks) {
        for (const tier of week.by_edge_tier) {
            const existing = tierStats.get(tier.tier) || { wins: 0, losses: 0 };
            tierStats.set(tier.tier, {
                wins: existing.wins + tier.wins,
                losses: existing.losses + tier.losses,
            });
        }
    }

    // Convert to array and filter out tiers with no results
    const tiers = Array.from(tierStats.entries())
        .map(([tier, stats]) => ({
            tier,
            ...stats,
            total: stats.wins + stats.losses,
            winRate: stats.wins + stats.losses > 0
                ? (stats.wins / (stats.wins + stats.losses)) * 100
                : 0,
        }))
        .filter((t) => t.total > 0);

    if (tiers.length === 0) {
        return null;
    }

    return (
        <div className="mt-3 px-3 py-2 rounded-lg bg-zinc-900/30 border border-zinc-800/50">
            <div className="text-xs font-medium text-zinc-500 mb-2">By Edge Tier</div>
            <div className="space-y-1">
                {tiers.map((tier) => (
                    <div
                        key={tier.tier}
                        className="flex items-center justify-between text-xs"
                    >
                        <span className="text-zinc-400">{tier.tier}</span>
                        <div className="flex items-center gap-2">
                            <span className="text-zinc-300">
                                {tier.wins}-{tier.losses}
                            </span>
                            <span
                                className={cn(
                                    "font-medium",
                                    tier.winRate >= 60
                                        ? "text-emerald-400"
                                        : tier.winRate >= 50
                                            ? "text-yellow-400"
                                            : "text-red-400"
                                )}
                            >
                                {tier.winRate.toFixed(0)}%
                            </span>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}
