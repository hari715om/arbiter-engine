import { useEffect, useState } from 'react'

interface Task {
    id: string
    status: string
    priority: number
    compute_cost: number
    assigned_worker: string | null
    estimated_duration: number
    deadline: number
    created_at: string | null
    resource_type: string
}

interface LiveEvent {
    event_type?: string
    type?: string
    task_id?: string
    worker_id?: string
    timestamp?: number
    [key: string]: unknown
}

interface Props {
    apiBase: string
    events: LiveEvent[]
    className?: string
}

function statusClass(status: string): string {
    return `status-badge status-${status.toLowerCase()}`
}

export default function TaskTimeline({ apiBase, events, className = '' }: Props) {
    const [tasks, setTasks] = useState<Task[]>([])
    const [view, setView] = useState<'tasks' | 'events'>('tasks')

    useEffect(() => {
        const poll = async () => {
            try {
                const r = await fetch(`${apiBase}/tasks?limit=50`)
                if (r.ok) setTasks(await r.json())
            } catch { /* offline */ }
        }
        poll()
        const id = setInterval(poll, 5000)
        return () => clearInterval(id)
    }, [apiBase])

    const evtColor = (type: string): string => {
        const t = type?.toUpperCase() || ''
        if (t.includes('COMPLET')) return 'var(--accent-green)'
        if (t.includes('FAIL')) return 'var(--accent-red)'
        if (t.includes('ASSIGN')) return 'var(--accent-blue)'
        if (t.includes('CREATE')) return 'var(--accent-purple)'
        return 'var(--text-muted)'
    }

    return (
        <div className={className} style={{ display: 'flex', flexDirection: 'column' }}>
            <div className="card-header">
                <span className="card-icon">📋</span>
                <span className="card-title">
                    {view === 'tasks' ? 'Task Queue' : 'Live Events'}
                </span>
                <div style={{ marginLeft: 'auto', display: 'flex', gap: 6 }}>
                    <button
                        onClick={() => setView('tasks')}
                        style={{
                            padding: '4px 12px', borderRadius: 4, border: '1px solid',
                            borderColor: view === 'tasks' ? 'var(--accent-blue)' : 'var(--border)',
                            background: view === 'tasks' ? 'rgba(99,179,237,0.12)' : 'transparent',
                            color: view === 'tasks' ? 'var(--accent-blue)' : 'var(--text-secondary)',
                            cursor: 'pointer', fontSize: 12,
                        }}
                    >Tasks</button>
                    <button
                        onClick={() => setView('events')}
                        style={{
                            padding: '4px 12px', borderRadius: 4, border: '1px solid',
                            borderColor: view === 'events' ? 'var(--accent-blue)' : 'var(--border)',
                            background: view === 'events' ? 'rgba(99,179,237,0.12)' : 'transparent',
                            color: view === 'events' ? 'var(--accent-blue)' : 'var(--text-secondary)',
                            cursor: 'pointer', fontSize: 12,
                        }}
                    >Events
                        {events.length > 0 && (
                            <span style={{
                                marginLeft: 6, background: 'var(--accent-blue)', color: '#000',
                                borderRadius: 10, padding: '1px 6px', fontSize: 10, fontWeight: 700,
                            }}>{Math.min(events.length, 99)}</span>
                        )}
                    </button>
                </div>
            </div>

            <div className="timeline-scroll" style={{ flex: 1 }}>
                {view === 'tasks' ? (
                    tasks.length === 0 ? (
                        <div className="empty-state">No tasks yet — POST to /tasks to add some</div>
                    ) : (
                        <table className="arbiter-table">
                            <thead>
                                <tr>
                                    <th>ID</th>
                                    <th>Status</th>
                                    <th>Priority</th>
                                    <th>Cost</th>
                                    <th>Worker</th>
                                    <th>Resource</th>
                                </tr>
                            </thead>
                            <tbody>
                                {tasks.map(t => (
                                    <tr key={t.id}>
                                        <td title={t.id}>{t.id.length > 16 ? t.id.slice(0, 14) + '…' : t.id}</td>
                                        <td><span className={statusClass(t.status)}>{t.status}</span></td>
                                        <td style={{ color: t.priority >= 8 ? 'var(--accent-red)' : t.priority >= 5 ? 'var(--accent-amber)' : 'var(--text-secondary)' }}>
                                            {t.priority}
                                        </td>
                                        <td>{t.compute_cost.toFixed(1)}</td>
                                        <td style={{ color: 'var(--text-secondary)' }}>{t.assigned_worker ?? '—'}</td>
                                        <td><span style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>{t.resource_type}</span></td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    )
                ) : (
                    events.length === 0 ? (
                        <div className="empty-state">Waiting for live events…</div>
                    ) : (
                        <table className="arbiter-table">
                            <thead>
                                <tr>
                                    <th>Event</th>
                                    <th>Task</th>
                                    <th>Worker</th>
                                    <th>Time</th>
                                </tr>
                            </thead>
                            <tbody>
                                {events.slice(0, 100).map((e, i) => {
                                    const evType = (e.event_type || e.type || 'UNKNOWN') as string
                                    return (
                                        <tr key={i}>
                                            <td style={{ color: evtColor(evType), fontWeight: 600 }}>{evType}</td>
                                            <td>{e.task_id ?? '—'}</td>
                                            <td style={{ color: 'var(--text-secondary)' }}>{e.worker_id ?? '—'}</td>
                                            <td style={{ color: 'var(--text-muted)' }}>
                                                {e.timestamp ? new Date(e.timestamp * 1000).toLocaleTimeString() : '—'}
                                            </td>
                                        </tr>
                                    )
                                })}
                            </tbody>
                        </table>
                    )
                )}
            </div>
        </div>
    )
}
