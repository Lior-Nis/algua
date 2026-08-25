import { useCallback, useEffect, useState } from 'react'

/** Error thrown by getJSON on a non-2xx response, carrying the backend envelope's code. */
export class ApiError extends Error {
  readonly status: number
  readonly code: string | null

  constructor(status: number, code: string | null, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.code = code
  }
}

export async function getJSON<T>(url: string): Promise<T> {
  const res = await fetch(url, { headers: { accept: 'application/json' } })
  if (!res.ok) {
    let code: string | null = null
    let message = `HTTP ${res.status}`
    try {
      const body = (await res.json()) as { code?: string; error?: string } | null
      if (body && typeof body === 'object') {
        code = typeof body.code === 'string' ? body.code : null
        if (typeof body.error === 'string' && body.error) message = body.error
      }
    } catch {
      // non-JSON error body — keep the HTTP status message
    }
    throw new ApiError(res.status, code, message)
  }
  return (await res.json()) as T
}

export interface RunsQuery {
  kind?: string
  strategy?: string
  family?: string
  sort?: string
  limit?: number
}

/** Builds the `/api/runs` query string. Centralizes the encoding so every run-ledger view (the
 * IS-vs-OOS scatter here; the ranked run list and trial distribution in later tasks) constructs
 * the same URL shape instead of each hand-rolling its own `encodeURIComponent` calls. */
export function runsUrl(query: RunsQuery = {}): string {
  const params = new URLSearchParams()
  if (query.kind) params.set('kind', query.kind)
  if (query.strategy) params.set('strategy', query.strategy)
  if (query.family) params.set('family', query.family)
  if (query.sort) params.set('sort', query.sort)
  if (query.limit !== undefined) params.set('limit', String(query.limit))
  const qs = params.toString()
  return qs ? `/api/runs?${qs}` : '/api/runs'
}

/** Format an ISO timestamp as a local HH:MM:SS for the header / banners. */
export function formatTimestamp(iso: string): string {
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleTimeString([], { hour12: false })
}

interface CacheEntry {
  at: number
  data: unknown
}

/** Module-level cache: survives route changes, shared across hook instances. */
const cache = new Map<string, CacheEntry>()

/** Module-level in-flight dedup: concurrent consumers of one URL (or StrictMode double
 * effects) share a single HTTP request instead of racing the cache. */
const inFlightByUrl = new Map<string, Promise<unknown>>()

function fetchShared<T>(url: string): Promise<T> {
  const existing = inFlightByUrl.get(url)
  if (existing) return existing as Promise<T>
  const p = getJSON<T>(url).finally(() => inFlightByUrl.delete(url))
  inFlightByUrl.set(url, p)
  return p
}

export interface UseFetchResult<T> {
  data: T | undefined
  error: Error | null
  loading: boolean
  refetch: () => void
}

/**
 * Tiny stale-while-revalidate fetch hook.
 * - Serves the cached payload immediately, revalidating in the background when
 *   the entry is older than ttlMs.
 * - Refetches when the tab becomes visible again (the phone-unlock case).
 */
export function useFetch<T>(url: string, opts: { ttlMs?: number } = {}): UseFetchResult<T> {
  const ttlMs = opts.ttlMs ?? 30_000
  const initial = cache.get(url)
  const [data, setData] = useState<T | undefined>(initial ? (initial.data as T) : undefined)
  const [error, setError] = useState<Error | null>(null)
  const [loading, setLoading] = useState(!initial)

  const load = useCallback(async () => {
    if (!cache.has(url)) setLoading(true)
    try {
      const fresh = await fetchShared<T>(url)
      cache.set(url, { at: Date.now(), data: fresh })
      setData(fresh)
      setError(null)
    } catch (e) {
      setError(e instanceof Error ? e : new Error(String(e)))
    } finally {
      setLoading(false)
    }
  }, [url])

  useEffect(() => {
    const hit = cache.get(url)
    if (hit) {
      setData(hit.data as T)
      setError(null)
      setLoading(false)
      if (Date.now() - hit.at >= ttlMs) void load()
    } else {
      setData(undefined)
      void load()
    }
    const onVisibility = () => {
      if (document.visibilityState === 'visible') void load()
    }
    document.addEventListener('visibilitychange', onVisibility)
    return () => document.removeEventListener('visibilitychange', onVisibility)
  }, [url, ttlMs, load])

  const refetch = useCallback(() => {
    void load()
  }, [load])

  return { data, error, loading, refetch }
}
