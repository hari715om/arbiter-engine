import { useEffect, useState } from 'react'
import { XAxis, YAxis, Tooltip, ResponsiveContainer, LineChart, Line } from 'recharts'

interface Metrics {
    total_tasks: number
    completed: number
    failed: number
    pending: number
    running: number
    queue_depth: number
    avg_latency: number | null
    sla_violation_rate: number | null
    worker_count: number
    active_workers: number
}

type MetricHistory = Array<{ time: string; completed: number; failed: number; pending: number }>

interface Props {
    apiBase: string
    className?: string
}

const CustomTooltip = ({ active, payload, label }: any) => {
    if (!active || !payload?.length) return null
    return (
        <div className="custom-tooltip">
            <div style={{ color: 'var(--text-secondary)', marginBottom: 4 }}>{label}</div>
            {payload.map((p: any) => (
                <div key={p.dataKey} style={{ color: p.color }}>
                    {p.dataKey}: {p.value}
                </div>
            ))}
        </div>
    )
}

export default function MetricsPanel({ apiBase, className = '' }: Props) {
    const [metrics, setMetrics] = useState<Metrics | null>(null)
    const [history, setHistory] = useState<MetricHistory>([])

    useEffect(() => {
        const poll = async () => {
            try {
                const r = await fetch(`${apiBase}/metrics`)
                if (!r.ok) return
                const m: Metrics = await r.json()
                setMetrics(m)
                const now = new Date().toLocaleTimeString('en', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })
                setHistory(h => [...h, { time: now, completed: m.completed, failed: m.failed, pending: m.pending }].slice(-20))
            } catch { /* offline */ }
        }
        poll()
        const id = setInterval(poll, 5000)
        return () => clearInterval(id)
    }, [apiBase])

    return (
        <div className={className}>
            <div className="card-header">
                <span className="card-icon">📊</span>
                <span className="card-title">Metrics</span>
            </div>

            {!metrics ? (
                <div className="card-body">
                    <div className="skeleton" style={{ width: '60%' }} />
                    <div className="skeleton" />
                    <div className="skeleton" style={{ width: '80%' }} />
                </div>
            ) : (
                <>
                    <div className="stat-grid">
                        <div className="stat-tile">
                            <span className="label">Total</span>
                            <span className="value color-blue">{metrics.total_tasks}</span>
                        </div>
                        <div className="stat-tile">
                            <span className="label">Completed</span>
                            <span className="value color-green">{metrics.completed}</span>
                        </div>
                        <div className="stat-tile">
                            <span className="label">Failed</span>
                            <span className="value color-red">{metrics.failed}</span>
                        </div>
                        <div className="stat-tile">
                            <span className="label">Running</span>
                            <span className="value color-amber">{metrics.running}</span>
                        </div>
                        <div className="stat-tile">
                            <span className="label">Queue depth</span>
                            <span className="value color-purple">{metrics.queue_depth}</span>
                        </div>
                        <div className="stat-tile">
                            <span className="label">SLA Viols</span>
                            <span className="value color-red">
                                {metrics.sla_violation_rate != null ? `${(metrics.sla_violation_rate * 100).toFixed(1)}%` : '—'}
                            </span>
                        </div>
                        <div className="stat-tile">
                            <span className="label">Avg Latency</span>
                            <span className="value color-teal">
                                {metrics.avg_latency != null ? `${metrics.avg_latency.toFixed(1)}s` : '—'}
                            </span>
                        </div>
                        <div className="stat-tile">
                            <span className="label">Workers</span>
                            <span className="value color-blue">{metrics.active_workers}/{metrics.worker_count}</span>
                        </div>
                    </div>

                    {history.length > 2 && (
                        <div className="card-body" style={{ paddingTop: 0 }}>
                            <div style={{ color: 'var(--text-muted)', fontSize: 11, textTransform: 'uppercase', marginBottom: 8 }}>Task counts over time</div>
                            <ResponsiveContainer width="100%" height={120}>
                                <LineChart data={history}>
                                    <XAxis dataKey="time" tick={{ fontSize: 10, fill: 'var(--text-muted)' }} interval="preserveStartEnd" />
                                    <YAxis tick={{ fontSize: 10, fill: 'var(--text-muted)' }} width={28} />
                                    <Tooltip content={<CustomTooltip />} />
                                    <Line type="monotone" dataKey="completed" stroke="var(--accent-green)" dot={false} strokeWidth={2} />
                                    <Line type="monotone" dataKey="failed" stroke="var(--accent-red)" dot={false} strokeWidth={2} />
                                    <Line type="monotone" dataKey="pending" stroke="var(--accent-amber)" dot={false} strokeWidth={2} />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    )}
                </>
            )}
        </div>
    )
}
