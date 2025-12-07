"use client";

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";

export default function SettingsPage() {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-zinc-100">Settings</h1>
        <p className="text-zinc-400 mt-1">Configure your betting parameters and preferences</p>
      </div>

      {/* Betting Parameters */}
      <Card className="bg-zinc-900 border-zinc-800">
        <CardHeader>
          <CardTitle className="text-zinc-100">Betting Parameters</CardTitle>
          <CardDescription className="text-zinc-500">
            Configure how value bets are calculated and sized
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="grid md:grid-cols-2 gap-6">
            <div className="space-y-2">
              <Label className="text-zinc-400">Min Edge Threshold (%)</Label>
              <Input
                type="number"
                defaultValue={5}
                min={0}
                max={50}
                step={0.5}
                className="bg-zinc-800 border-zinc-700 text-zinc-100"
              />
              <p className="text-xs text-zinc-500">
                Only show bets with edge above this percentage
              </p>
            </div>

            <div className="space-y-2">
              <Label className="text-zinc-400">Kelly Fraction</Label>
              <Input
                type="number"
                defaultValue={0.25}
                min={0}
                max={1}
                step={0.05}
                className="bg-zinc-800 border-zinc-700 text-zinc-100"
              />
              <p className="text-xs text-zinc-500">
                Fraction of Kelly criterion to use for bet sizing
              </p>
            </div>

            <div className="space-y-2">
              <Label className="text-zinc-400">Max Stake (%)</Label>
              <Input
                type="number"
                defaultValue={2}
                min={0}
                max={10}
                step={0.5}
                className="bg-zinc-800 border-zinc-700 text-zinc-100"
              />
              <p className="text-xs text-zinc-500">
                Maximum percentage of bankroll per bet
              </p>
            </div>

            <div className="space-y-2">
              <Label className="text-zinc-400">Bankroll ($)</Label>
              <Input
                type="number"
                defaultValue={1000}
                min={0}
                step={100}
                className="bg-zinc-800 border-zinc-700 text-zinc-100"
              />
              <p className="text-xs text-zinc-500">
                Your total betting bankroll
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Model Settings */}
      <Card className="bg-zinc-900 border-zinc-800">
        <CardHeader>
          <CardTitle className="text-zinc-100">Model Adjustments</CardTitle>
          <CardDescription className="text-zinc-500">
            Enable or disable model features
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <Label className="text-zinc-100">Defense Multipliers</Label>
              <p className="text-xs text-zinc-500">
                Adjust predictions based on defense vs position stats
              </p>
            </div>
            <Switch defaultChecked />
          </div>

          <Separator className="bg-zinc-800" />

          <div className="flex items-center justify-between">
            <div>
              <Label className="text-zinc-100">Weather Adjustments</Label>
              <p className="text-xs text-zinc-500">
                Factor in weather conditions for outdoor games
              </p>
            </div>
            <Switch defaultChecked />
          </div>

          <Separator className="bg-zinc-800" />

          <div className="flex items-center justify-between">
            <div>
              <Label className="text-zinc-100">Injury Weighting</Label>
              <p className="text-xs text-zinc-500">
                Adjust projections based on injury status
              </p>
            </div>
            <Switch defaultChecked />
          </div>
        </CardContent>
      </Card>

      {/* Display Preferences */}
      <Card className="bg-zinc-900 border-zinc-800">
        <CardHeader>
          <CardTitle className="text-zinc-100">Display Preferences</CardTitle>
          <CardDescription className="text-zinc-500">
            Customize how data is displayed
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <Label className="text-zinc-100">Best Line Only</Label>
              <p className="text-xs text-zinc-500">
                Show only the best sportsbook line per player
              </p>
            </div>
            <Switch defaultChecked />
          </div>

          <Separator className="bg-zinc-800" />

          <div className="flex items-center justify-between">
            <div>
              <Label className="text-zinc-100">Show Synthetic Odds</Label>
              <p className="text-xs text-zinc-500">
                Include simulated odds when real odds unavailable
              </p>
            </div>
            <Switch />
          </div>
        </CardContent>
      </Card>

      {/* Save Button */}
      <div className="flex justify-end">
        <Button className="bg-blue-600 hover:bg-blue-700 text-white">
          Save Settings
        </Button>
      </div>
    </div>
  );
}

