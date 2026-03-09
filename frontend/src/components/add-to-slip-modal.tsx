"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { cn } from "@/lib/utils";
import { recordBet } from "@/lib/api";
import type { ValueBet } from "@/lib/types";
import { X } from "lucide-react";

interface AddToSlipModalProps {
  bet: ValueBet;
  season: number;
  week: number;
  onClose: () => void;
  onSuccess?: () => void;
}

export function AddToSlipModal({
  bet,
  season,
  week,
  onClose,
  onSuccess,
}: AddToSlipModalProps) {
  const [stake, setStake] = useState<string>(
    bet.stake > 0 ? bet.stake.toFixed(2) : "1.00"
  );
  const [sportsbook, setSportsbook] = useState<string>(bet.sportsbook ?? "");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const stakeNum = parseFloat(stake);
  const isValidStake = !isNaN(stakeNum) && stakeNum > 0;

  const handleSubmit = async () => {
    if (!isValidStake) return;
    setSubmitting(true);
    setError(null);
    try {
      await recordBet({
        season,
        week,
        player_id: bet.player_id,
        market: bet.market,
        sportsbook: sportsbook || bet.sportsbook,
        line: bet.line,
        price: bet.price,
        stake: stakeNum,
        edge_at_placement: bet.edge_percentage,
      });
      onSuccess?.();
      onClose();
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to record bet. Are you signed in?"
      );
    } finally {
      setSubmitting(false);
    }
  };

  const potentialProfit =
    isValidStake
      ? bet.price > 0
        ? (stakeNum * bet.price) / 100
        : (stakeNum * 100) / Math.abs(bet.price)
      : 0;

  return (
    /* Backdrop */
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
      onClick={onClose}
    >
      {/* Modal */}
      <div
        className="relative w-full max-w-sm mx-4 rounded-xl bg-[#0d1220] border border-slate-700/60 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-start justify-between p-5 border-b border-slate-800/50">
          <div>
            <h2 className="text-sm font-semibold text-slate-100">Add to Bet Slip</h2>
            <p className="text-xs text-slate-500 mt-0.5">
              {bet.player_name ?? bet.player_id} — {bet.market?.replace(/_/g, " ")}
            </p>
          </div>
          <button
            onClick={onClose}
            className="text-slate-600 hover:text-slate-300 transition-colors ml-3 mt-0.5"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Bet summary */}
        <div className="px-5 pt-4 pb-2 grid grid-cols-3 gap-3 text-center">
          <div className="bg-slate-800/30 rounded-lg py-2.5">
            <p className="text-[10px] text-slate-600 uppercase tracking-wider mb-1">Line</p>
            <p className="text-base font-bold text-slate-200 font-[family-name:var(--font-jetbrains)]">
              O {bet.line?.toFixed(1)}
            </p>
          </div>
          <div className="bg-slate-800/30 rounded-lg py-2.5">
            <p className="text-[10px] text-slate-600 uppercase tracking-wider mb-1">Price</p>
            <p className="text-base font-bold text-slate-200 font-[family-name:var(--font-jetbrains)]">
              {bet.price > 0 ? `+${bet.price}` : bet.price}
            </p>
          </div>
          <div className="bg-slate-800/30 rounded-lg py-2.5">
            <p className="text-[10px] text-slate-600 uppercase tracking-wider mb-1">Edge</p>
            <p
              className={cn(
                "text-base font-bold font-[family-name:var(--font-jetbrains)]",
                bet.edge_percentage >= 0.1
                  ? "text-emerald-400"
                  : bet.edge_percentage >= 0.05
                  ? "text-amber-400"
                  : "text-slate-400"
              )}
            >
              {(bet.edge_percentage * 100).toFixed(1)}%
            </p>
          </div>
        </div>

        {/* Form */}
        <div className="px-5 pb-5 space-y-4 mt-2">
          {/* Sportsbook */}
          <div className="space-y-1.5">
            <Label className="text-xs text-slate-500 uppercase tracking-wider">
              Sportsbook
            </Label>
            <Input
              value={sportsbook}
              onChange={(e) => setSportsbook(e.target.value)}
              placeholder="e.g. DraftKings"
              className="bg-slate-900/60 border-slate-700 text-slate-200 text-sm h-9 placeholder:text-slate-700"
            />
          </div>

          {/* Stake */}
          <div className="space-y-1.5">
            <Label className="text-xs text-slate-500 uppercase tracking-wider">
              Stake (units)
            </Label>
            <Input
              type="number"
              min="0.01"
              step="0.1"
              value={stake}
              onChange={(e) => setStake(e.target.value)}
              className="bg-slate-900/60 border-slate-700 text-slate-200 text-sm h-9"
            />
          </div>

          {/* Profit estimate */}
          {isValidStake && (
            <div className="flex items-center justify-between text-xs px-1">
              <span className="text-slate-600">Est. profit if win</span>
              <span className="text-emerald-400 font-mono font-medium">
                +{potentialProfit.toFixed(2)}u
              </span>
            </div>
          )}

          {/* Error */}
          {error && (
            <p className="text-xs text-red-400 bg-red-900/10 border border-red-800/30 rounded-md px-3 py-2">
              {error}
            </p>
          )}

          {/* Actions */}
          <div className="flex gap-2 pt-1">
            <Button
              variant="ghost"
              size="sm"
              onClick={onClose}
              className="flex-1 text-slate-500 hover:text-slate-300 hover:bg-slate-800/50"
            >
              Cancel
            </Button>
            <Button
              size="sm"
              onClick={handleSubmit}
              disabled={submitting || !isValidStake}
              className="flex-1 bg-blue-600 hover:bg-blue-500 text-white disabled:opacity-50"
            >
              {submitting ? "Recording…" : "Add to Slip"}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
