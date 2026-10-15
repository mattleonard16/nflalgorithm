"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { useAuth } from "@/lib/auth-context";
import { getUserPreferences, updateUserBankroll, updateUserPreferences } from "@/lib/api";
import type { UserPreferences } from "@/lib/types";

// Matches the UserPreferences defaults in api/auth.py, used until the saved
// row loads (or when the account has none yet).
const DEFAULT_PREFERENCES: UserPreferences = {
  default_min_edge: 0.05,
  default_kelly_fraction: 0.25,
  default_max_stake: 0.02,
  best_line_only: true,
  show_synthetic_odds: false,
  defense_multipliers: true,
  weather_adjustments: true,
  injury_weighting: true,
  preferred_sportsbooks: null,
  preferred_markets: null,
};

// Numeric inputs are held as strings so a half-typed value ("0.") is not
// coerced mid-edit; they are parsed and range-checked on save.
interface NumericFields {
  minEdgePct: string;
  kellyFraction: string;
  maxStakePct: string;
  bankroll: string;
}

function toNumericFields(prefs: UserPreferences, bankroll: number): NumericFields {
  return {
    minEdgePct: String(+(prefs.default_min_edge * 100).toFixed(2)),
    kellyFraction: String(prefs.default_kelly_fraction),
    maxStakePct: String(+(prefs.default_max_stake * 100).toFixed(2)),
    bankroll: String(bankroll),
  };
}

function parseInRange(value: string, min: number, max: number): number | null {
  if (value.trim() === "") return null;
  const n = Number(value);
  return Number.isFinite(n) && n >= min && n <= max ? n : null;
}

type Status = { kind: "success" | "error"; message: string } | null;

export default function SettingsPage() {
  const { user, loading: authLoading, refreshUser } = useAuth();
  const [prefs, setPrefs] = useState<UserPreferences>(DEFAULT_PREFERENCES);
  const [fields, setFields] = useState<NumericFields>(() =>
    toNumericFields(DEFAULT_PREFERENCES, 1000)
  );
  const [loadingPrefs, setLoadingPrefs] = useState(false);
  const [saving, setSaving] = useState(false);
  const [status, setStatus] = useState<Status>(null);

  const userId = user?.id ?? null;
  const userBankroll = user?.bankroll ?? null;

  // Load the saved preferences and bankroll for the signed-in user. A bankroll
  // save refreshes the user, which reloads the (just-saved) values here.
  useEffect(() => {
    if (!userId) return;
    let cancelled = false;

    async function load() {
      setLoadingPrefs(true);
      try {
        const saved = await getUserPreferences();
        if (cancelled) return;
        const loaded = saved ?? DEFAULT_PREFERENCES;
        setPrefs(loaded);
        setFields(toNumericFields(loaded, userBankroll ?? 0));
      } catch (err) {
        if (!cancelled) {
          setStatus({
            kind: "error",
            message: err instanceof Error ? err.message : "Failed to load settings",
          });
        }
      } finally {
        if (!cancelled) setLoadingPrefs(false);
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, [userId, userBankroll]);

  const setField = (key: keyof NumericFields) => (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setFields((current) => ({ ...current, [key]: value }));
    setStatus(null);
  };

  const setToggle = (key: keyof UserPreferences) => (checked: boolean) => {
    setPrefs((current) => ({ ...current, [key]: checked }));
    setStatus(null);
  };

  const handleSave = async () => {
    if (!user) return;
    const minEdgePct = parseInRange(fields.minEdgePct, 0, 50);
    const kellyFraction = parseInRange(fields.kellyFraction, 0, 1);
    const maxStakePct = parseInRange(fields.maxStakePct, 0, 10);
    const bankroll = parseInRange(fields.bankroll, 0, Number.MAX_SAFE_INTEGER);

    if (minEdgePct === null || kellyFraction === null || maxStakePct === null || bankroll === null) {
      const problems: string[] = [];
      if (minEdgePct === null) problems.push("Min Edge must be between 0 and 50.");
      if (kellyFraction === null) problems.push("Kelly Fraction must be between 0 and 1.");
      if (maxStakePct === null) problems.push("Max Stake must be between 0 and 10.");
      if (bankroll === null) problems.push("Bankroll must be zero or more.");
      setStatus({ kind: "error", message: problems.join(" ") });
      return;
    }

    const next: UserPreferences = {
      ...prefs,
      default_min_edge: minEdgePct / 100,
      default_kelly_fraction: kellyFraction,
      default_max_stake: maxStakePct / 100,
    };

    setSaving(true);
    setStatus(null);
    try {
      await updateUserPreferences(next);
      setPrefs(next);
      if (bankroll !== user.bankroll) {
        await updateUserBankroll(bankroll);
        await refreshUser();
      }
      setStatus({ kind: "success", message: "Settings saved." });
    } catch (err) {
      setStatus({
        kind: "error",
        message: err instanceof Error ? err.message : "Failed to save settings",
      });
    } finally {
      setSaving(false);
    }
  };

  const disabled = !user || loadingPrefs || saving;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-4xl font-bold text-slate-100 tracking-tight font-display uppercase">
          Settings
        </h1>
        <p className="text-sm text-slate-500 mt-0.5">
          Edge floor, Kelly fraction, and display
        </p>
      </div>

      {!authLoading && !user && (
        <div className="rounded-lg border border-amber-800/40 bg-amber-900/10 p-4 text-sm text-amber-400">
          <Link href="/login" className="underline hover:text-amber-300">
            Sign in
          </Link>{" "}
          to load and save your settings.
        </div>
      )}

      {/* Betting Parameters */}
      <Card className="bg-[#111827] border-slate-800">
        <CardHeader>
          <CardTitle className="text-slate-100">Betting Parameters</CardTitle>
          <CardDescription className="text-slate-500">
            Configure how value bets are calculated and sized
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="grid md:grid-cols-2 gap-6">
            <div className="space-y-2">
              <Label htmlFor="min-edge" className="text-slate-400">Min Edge Threshold (%)</Label>
              <Input
                id="min-edge"
                type="number"
                value={fields.minEdgePct}
                onChange={setField("minEdgePct")}
                disabled={disabled}
                min={0}
                max={50}
                step={0.5}
                className="bg-[#0d1220] border-slate-700 text-slate-100"
              />
              <p className="text-xs text-slate-500">
                Only show bets with edge above this percentage
              </p>
            </div>

            <div className="space-y-2">
              <Label htmlFor="kelly-fraction" className="text-slate-400">Kelly Fraction</Label>
              <Input
                id="kelly-fraction"
                type="number"
                value={fields.kellyFraction}
                onChange={setField("kellyFraction")}
                disabled={disabled}
                min={0}
                max={1}
                step={0.05}
                className="bg-[#0d1220] border-slate-700 text-slate-100"
              />
              <p className="text-xs text-slate-500">
                Fraction of Kelly criterion to use for bet sizing
              </p>
            </div>

            <div className="space-y-2">
              <Label htmlFor="max-stake" className="text-slate-400">Max Stake (%)</Label>
              <Input
                id="max-stake"
                type="number"
                value={fields.maxStakePct}
                onChange={setField("maxStakePct")}
                disabled={disabled}
                min={0}
                max={10}
                step={0.5}
                className="bg-[#0d1220] border-slate-700 text-slate-100"
              />
              <p className="text-xs text-slate-500">
                Maximum percentage of bankroll per bet
              </p>
            </div>

            <div className="space-y-2">
              <Label htmlFor="bankroll" className="text-slate-400">Bankroll ($)</Label>
              <Input
                id="bankroll"
                type="number"
                value={fields.bankroll}
                onChange={setField("bankroll")}
                disabled={disabled}
                min={0}
                step={100}
                className="bg-[#0d1220] border-slate-700 text-slate-100"
              />
              <p className="text-xs text-slate-500">
                Your total betting bankroll
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Model Settings */}
      <Card className="bg-[#111827] border-slate-800">
        <CardHeader>
          <CardTitle className="text-slate-100">Model Adjustments</CardTitle>
          <CardDescription className="text-slate-500">
            Enable or disable model features
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <Label className="text-slate-100">Defense Multipliers</Label>
              <p className="text-xs text-slate-500">
                Adjust predictions based on defense vs position stats
              </p>
            </div>
            <Switch
              checked={prefs.defense_multipliers}
              onCheckedChange={setToggle("defense_multipliers")}
              disabled={disabled}
            />
          </div>

          <Separator className="bg-slate-800" />

          <div className="flex items-center justify-between">
            <div>
              <Label className="text-slate-100">Weather Adjustments</Label>
              <p className="text-xs text-slate-500">
                Factor in weather conditions for outdoor games
              </p>
            </div>
            <Switch
              checked={prefs.weather_adjustments}
              onCheckedChange={setToggle("weather_adjustments")}
              disabled={disabled}
            />
          </div>

          <Separator className="bg-slate-800" />

          <div className="flex items-center justify-between">
            <div>
              <Label className="text-slate-100">Injury Weighting</Label>
              <p className="text-xs text-slate-500">
                Adjust projections based on injury status
              </p>
            </div>
            <Switch
              checked={prefs.injury_weighting}
              onCheckedChange={setToggle("injury_weighting")}
              disabled={disabled}
            />
          </div>
        </CardContent>
      </Card>

      {/* Display Preferences */}
      <Card className="bg-[#111827] border-slate-800">
        <CardHeader>
          <CardTitle className="text-slate-100">Display Preferences</CardTitle>
          <CardDescription className="text-slate-500">
            Customize how data is displayed
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <Label className="text-slate-100">Best Line Only</Label>
              <p className="text-xs text-slate-500">
                Show only the best sportsbook line per player
              </p>
            </div>
            <Switch
              checked={prefs.best_line_only}
              onCheckedChange={setToggle("best_line_only")}
              disabled={disabled}
            />
          </div>

          <Separator className="bg-slate-800" />

          <div className="flex items-center justify-between">
            <div>
              <Label className="text-slate-100">Show Synthetic Odds</Label>
              <p className="text-xs text-slate-500">
                Include simulated odds when real odds unavailable
              </p>
            </div>
            <Switch
              checked={prefs.show_synthetic_odds}
              onCheckedChange={setToggle("show_synthetic_odds")}
              disabled={disabled}
            />
          </div>
        </CardContent>
      </Card>

      {/* Save Button */}
      <div className="flex items-center justify-end gap-3">
        {status && (
          <p
            role={status.kind === "error" ? "alert" : "status"}
            className={
              status.kind === "error" ? "text-sm text-red-400" : "text-sm text-emerald-400"
            }
          >
            {status.message}
          </p>
        )}
        <Button
          onClick={handleSave}
          disabled={disabled}
          className="bg-primary text-primary-foreground hover:bg-amber-300 disabled:opacity-50"
        >
          {saving ? "Saving…" : "Save Settings"}
        </Button>
      </div>
    </div>
  );
}
