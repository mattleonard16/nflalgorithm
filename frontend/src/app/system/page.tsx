"use client";

import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { getHealth, ping } from "@/lib/api";
import type { HealthResponse } from "@/lib/types";
import { CheckCircle, XCircle, AlertCircle, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";

function StatusBadge({ status }: { status: "ACTIVE" | "MAINTENANCE" | "FRESH" | "STALE" }) {
  const variants: Record<string, { className: string; icon: React.ReactNode }> = {
    ACTIVE: {
      className: "bg-green-900/50 text-green-300 border-green-700",
      icon: <CheckCircle className="h-3 w-3" />,
    },
    FRESH: {
      className: "bg-green-900/50 text-green-300 border-green-700",
      icon: <CheckCircle className="h-3 w-3" />,
    },
    MAINTENANCE: {
      className: "bg-amber-900/50 text-amber-300 border-amber-700",
      icon: <AlertCircle className="h-3 w-3" />,
    },
    STALE: {
      className: "bg-red-900/50 text-red-300 border-red-700",
      icon: <XCircle className="h-3 w-3" />,
    },
  };

  const variant = variants[status] || variants.STALE;

  return (
    <Badge className={`${variant.className} gap-1`} variant="outline">
      {variant.icon}
      {status}
    </Badge>
  );
}

function StatusCard({
  title,
  value,
  status,
}: {
  title: string;
  value: string;
  status: "good" | "warn" | "error";
}) {
  const colors = {
    good: "text-green-400",
    warn: "text-amber-400",
    error: "text-red-400",
  };

  return (
    <Card className="bg-zinc-900 border-zinc-800">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium text-zinc-400">{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <div className={`text-xl font-bold ${colors[status]}`}>{value}</div>
      </CardContent>
    </Card>
  );
}

export default function SystemPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [apiStatus, setApiStatus] = useState<"checking" | "online" | "offline">("checking");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    setApiStatus("checking");

    try {
      const isOnline = await ping();
      setApiStatus(isOnline ? "online" : "offline");

      if (isOnline) {
        const healthData = await getHealth();
        setHealth(healthData);
      }
    } catch (err) {
      setApiStatus("offline");
      setError(err instanceof Error ? err.message : "Failed to fetch health data");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const formatAge = (minutes: number | null) => {
    if (minutes === null) return "Unknown";
    if (minutes < 1) return "< 1 min";
    if (minutes < 60) return `${minutes.toFixed(0)} min`;
    return `${(minutes / 60).toFixed(1)} hrs`;
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-start">
        <div>
          <h1 className="text-3xl font-bold text-zinc-100">System Health</h1>
          <p className="text-zinc-400 mt-1">Monitor system status and data freshness</p>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={fetchData}
          disabled={loading}
          className="border-zinc-700 text-zinc-300 hover:bg-zinc-800"
        >
          <RefreshCw className={`h-4 w-4 mr-2 ${loading ? "animate-spin" : ""}`} />
          Refresh
        </Button>
      </div>

      {error && (
        <div className="bg-red-900/20 border border-red-800 rounded-lg p-4 text-red-300">
          {error}
        </div>
      )}

      {/* Status Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatusCard
          title="API Status"
          value={apiStatus === "checking" ? "Checking..." : apiStatus === "online" ? "Online" : "Offline"}
          status={apiStatus === "online" ? "good" : apiStatus === "offline" ? "error" : "warn"}
        />
        <StatusCard
          title="System Status"
          value={health?.status || "Unknown"}
          status={health?.status === "ACTIVE" ? "good" : "warn"}
        />
        <StatusCard
          title="Database"
          value={health?.database || "Unknown"}
          status={health?.database === "Connected" ? "good" : "error"}
        />
        <StatusCard
          title="Last Update"
          value={
            health?.last_update
              ? new Date(health.last_update).toLocaleTimeString()
              : "Unknown"
          }
          status="good"
        />
      </div>

      {/* Feed Freshness */}
      <Card className="bg-zinc-900 border-zinc-800">
        <CardHeader>
          <CardTitle className="text-zinc-100">Data Feed Freshness</CardTitle>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="text-center py-8 text-zinc-400">Loading...</div>
          ) : health?.feeds && health.feeds.length > 0 ? (
            <Table>
              <TableHeader>
                <TableRow className="border-zinc-800">
                  <TableHead className="text-zinc-400">Feed</TableHead>
                  <TableHead className="text-zinc-400">Last Updated</TableHead>
                  <TableHead className="text-zinc-400">Age</TableHead>
                  <TableHead className="text-zinc-400">Status</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {health.feeds.map((feed) => (
                  <TableRow key={feed.feed} className="border-zinc-800">
                    <TableCell className="text-zinc-100 capitalize font-medium">
                      {feed.feed}
                    </TableCell>
                    <TableCell className="text-zinc-400">
                      {feed.as_of
                        ? new Date(feed.as_of).toLocaleString()
                        : "Never"}
                    </TableCell>
                    <TableCell className="text-zinc-400">
                      {formatAge(feed.age_minutes)}
                    </TableCell>
                    <TableCell>
                      <StatusBadge status={feed.status} />
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          ) : (
            <div className="text-center py-8 text-zinc-400">
              No feed freshness data available. Run the data pipeline to populate feeds.
            </div>
          )}
        </CardContent>
      </Card>

      {/* System Info */}
      <Card className="bg-zinc-900 border-zinc-800">
        <CardHeader>
          <CardTitle className="text-zinc-100">System Information</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-zinc-400">API Version</span>
                <span className="text-zinc-100">2.1.0</span>
              </div>
              <div className="flex justify-between">
                <span className="text-zinc-400">Frontend Version</span>
                <span className="text-zinc-100">2.1.0</span>
              </div>
              <div className="flex justify-between">
                <span className="text-zinc-400">Backend Framework</span>
                <span className="text-zinc-100">FastAPI</span>
              </div>
            </div>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-zinc-400">Model Version</span>
                <span className="text-zinc-100">weekly-latest</span>
              </div>
              <div className="flex justify-between">
                <span className="text-zinc-400">Database Backend</span>
                <span className="text-zinc-100">SQLite</span>
              </div>
              <div className="flex justify-between">
                <span className="text-zinc-400">Environment</span>
                <span className="text-zinc-100">Development</span>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

