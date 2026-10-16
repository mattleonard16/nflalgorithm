"use client";

import { createContext, useContext, useEffect, useState, ReactNode } from "react";
import type { User } from "./types";
import {
  AUTH_EXPIRED_EVENT,
  getCurrentUser,
  login as apiLogin,
  logout as apiLogout,
  register as apiRegister,
} from "./api";

// Mirrors PIPELINE_OPERATOR_TIERS in api/pipeline_router.py (default
// "operator,admin"). Override at build time when the backend's list changes.
const PIPELINE_OPERATOR_TIERS = new Set(
  (process.env.NEXT_PUBLIC_PIPELINE_OPERATOR_TIERS ?? "operator,admin")
    .split(",")
    .map((tier) => tier.trim().toLowerCase())
    .filter(Boolean)
);

/**
 * Whether the backend will let this user run or review the pipeline
 * (require_pipeline_operator). Client-side gating only hides controls that
 * would 403; the server remains the authority.
 */
export function isPipelineOperator(user: User | null): boolean {
  return !!user && PIPELINE_OPERATOR_TIERS.has(user.subscription_tier?.toLowerCase() ?? "");
}

interface AuthContextType {
  user: User | null;
  loading: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string, name?: string) => Promise<void>;
  logout: () => Promise<void>;
  refreshUser: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  const refreshUser = async () => {
    try {
      const currentUser = await getCurrentUser();
      setUser(currentUser);
    } catch {
      setUser(null);
    }
  };

  useEffect(() => {
    async function init() {
      try {
        await refreshUser();
      } finally {
        setLoading(false);
      }
    }
    init();
  }, []);

  const login = async (email: string, password: string) => {
    const result = await apiLogin(email, password);
    setUser(result.user);
  };

  const register = async (email: string, password: string, name?: string) => {
    const result = await apiRegister(email, password, name);
    setUser(result.user);
  };

  // A 401 from any authenticated call has already cleared the stored token;
  // drop the user so the UI stops rendering a signed-in state.
  useEffect(() => {
    const onExpired = () => setUser(null);
    window.addEventListener(AUTH_EXPIRED_EVENT, onExpired);
    return () => window.removeEventListener(AUTH_EXPIRED_EVENT, onExpired);
  }, []);

  const logout = async () => {
    try {
      await apiLogout();
    } catch {
      // apiLogout clears the local token even when the server call fails
      // (e.g. the session already expired), so signing out locally is correct.
    } finally {
      setUser(null);
    }
  };

  return (
    <AuthContext.Provider value={{ user, loading, login, register, logout, refreshUser }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}

