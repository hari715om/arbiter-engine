import { useEffect, useState } from 'react'

interface Worker {
    id: string
    status: string
    cpu_capacity: number
    memory_capacity: number
    speed_multiplier: number
    current_load: number
    supported_resources: string[]
}

interface Props {
    apiBase: string
    className?: string
}

function getUtilisation(worker: Worker): number {
    return worker.cpu_capacity > 0 ? worker.current_load / worker.cpu_capacity : 0
}

function utilColor(u: number): string {
    if (u < 0.3) return 'rgba(104,211,145,0.15)'
    if (u < 0.7) return 'rgba(246,173,85,0.18)'
    if (u < 0.9) return 'rgba(252,129,129,0.18)'
    return 'rgba(252,129,129,0.35)'
}

function utilTextColor(u: number): string {
    if (u < 0.3) return 'var(--accent-green)'
    if (u < 0.7) return 'var(--accent-amber)'
    return 'var(--accent-red)'
}

function statusColor(status: string): string {
    if (status === 'idle') return 'var(--accent-green)'
    if (status === 'busy') return 'var(--accent-amber)'
    return 'var(--text-muted)'
}

export default function WorkerHeatmap({ apiBase, className = '' }: Props) {
    const [workers, setWorkers] = useState<Worker[]>([])
    const [hover, setHover] = useState<string | null>(null)

    useEffect(() => {
        const poll = async () => {
            try {
                const r = await fetch(`${apiBase}/workers`)
                if (r.ok) setWorkers(await r.json())
            } catch { /* offline */ }
        }
        poll()
        const id = setInterval(poll, 2000)
        return () => clearInterval(id)
    }, [apiBase])

    return (
        <div className={className}>
            <div className="card-header">
                <span className="card-icon">🖥️</span>
                <span className="card-title">Workers</span>
                <span style={{ marginLeft: 'auto', fontSize: 12, color: 'var(--text-muted)' }}>
                    {workers.filter(w => w.status === 'busy').length} busy / {workers.length} total
                </span>
            </div>

            {workers.length === 0 ? (
                <div className="empty-state">No workers registered</div>
            ) : (
                <div className="worker-grid">
                    {workers.map(w => {
                        const u = getUtilisation(w)
                        const isHovered = hover === w.id
                        return (
                            <div
                                key={w.id}
                                className="worker-cell"
                                style={{ background: utilColor(u), borderColor: isHovered ? 'var(--border-glow)' : 'var(--border)' }}
                                onMouseEnter={() => setHover(w.id)}
                                onMouseLeave={() => setHover(null)}
                                title={`Status: ${w.status}\nCPU: ${w.current_load.toFixed(1)}/${w.cpu_capacity}\nMem: ${w.memory_capacity} GB\nSpeed: ${w.speed_multiplier}×\nResources: ${w.supported_resources.join(', ')}`}
                            >
                                <div className="worker-id">{w.id.replace('worker-', 'w-')}</div>
                                <div className="worker-load" style={{ color: utilTextColor(u) }}>
                                    {(u * 100).toFixed(0)}%
                                </div>
                                <div className="worker-cap">{w.current_load.toFixed(1)}/{w.cpu_capacity} CPU</div>
                                <div style={{ fontSize: 10, color: statusColor(w.status), marginTop: 4, textTransform: 'uppercase', fontWeight: 600 }}>
                                    {w.status}
                                </div>
                            </div>
                        )
                    })}
                </div>
            )}

            <div style={{ padding: '8px 16px', display: 'flex', gap: 16, fontSize: 11, color: 'var(--text-muted)', borderTop: '1px solid var(--border)' }}>
                <span>🟢 &lt;30% idle</span>
                <span>🟡 30-70% moderate</span>
                <span>🔴 &gt;70% high</span>
            </div>
        </div>
    )
}
