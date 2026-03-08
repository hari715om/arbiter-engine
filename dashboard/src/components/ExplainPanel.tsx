import { useState } from 'react'

interface AlternativeAssignment {
    worker_id: string
    score: number
    breakdown: Record<string, number>
}

interface ExplanationResponse {
    task_id: string
    worker_id: string | null
    scheduler_name: string
    total_score: number
    factors: Record<string, number>
    reasoning: string
    alternatives: AlternativeAssignment[]
}

interface Props {
    apiBase: string
    className?: string
}

export default function ExplainPanel({ apiBase, className = '' }: Props) {
    const [taskId, setTaskId] = useState('')
    const [loading, setLoading] = useState(false)
    const [result, setResult] = useState<ExplanationResponse | null>(null)
    const [error, setError] = useState<string | null>(null)

    const maxScore = result
        ? Math.max(...Object.values(result.factors), 0.001)
        : 1

    const handleExplain = async () => {
        if (!taskId.trim()) return
        setLoading(true)
        setError(null)
        setResult(null)
        try {
            const r = await fetch(`${apiBase}/tasks/${taskId.trim()}/explain`)
            if (!r.ok) {
                const body = await r.json().catch(() => null)
                setError(body?.detail ?? `HTTP ${r.status}`)
                return
            }
            setResult(await r.json())
        } catch (e: any) {
            setError(e.message ?? 'Network error')
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className={className} style={{ display: 'flex', flexDirection: 'column' }}>
            <div className="card-header">
                <span className="card-icon">🔍</span>
                <span className="card-title">Explain Assignment</span>
            </div>

            <div className="explain-input-row">
                <input
                    className="explain-input"
                    id="explain-task-id"
                    placeholder="Enter task ID…"
                    value={taskId}
                    onChange={e => setTaskId(e.target.value)}
                    onKeyDown={e => e.key === 'Enter' && handleExplain()}
                />
                <button className="btn btn-primary" onClick={handleExplain} disabled={loading || !taskId.trim()}>
                    {loading ? '…' : 'Explain'}
                </button>
            </div>

            <div className="explain-body" style={{ flex: 1, overflowY: 'auto' }}>
                {error && (
                    <div style={{ color: 'var(--accent-red)', fontSize: 12, padding: '8px 0' }}>
                        ✗ {error}
                    </div>
                )}

                {!result && !error && !loading && (
                    <div className="empty-state" style={{ paddingTop: 30 }}>
                        Enter a task ID to see why it was assigned to its worker
                    </div>
                )}

                {loading && (
                    <>
                        <div className="skeleton" />
                        <div className="skeleton" style={{ width: '80%' }} />
                        <div className="skeleton" style={{ width: '60%' }} />
                    </>
                )}

                {result && (
                    <>
                        <div style={{ marginBottom: 14 }}>
                            <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', fontSize: 12 }}>
                                <span style={{ color: 'var(--text-muted)' }}>Task</span>
                                <span style={{ color: 'var(--accent-blue)', fontFamily: 'var(--font-mono)' }}>{result.task_id}</span>
                                <span style={{ color: 'var(--text-muted)' }}>→</span>
                                <span style={{ color: 'var(--text-muted)' }}>Worker</span>
                                <span style={{ color: 'var(--accent-green)', fontFamily: 'var(--font-mono)' }}>{result.worker_id ?? '(unassigned)'}</span>
                            </div>
                            <div style={{ display: 'flex', gap: 12, marginTop: 6, fontSize: 12 }}>
                                <span style={{ color: 'var(--text-muted)' }}>Scheduler:</span>
                                <span style={{ color: 'var(--accent-purple)' }}>{result.scheduler_name}</span>
                                <span style={{ color: 'var(--text-muted)' }}>Score:</span>
                                <span style={{ color: 'var(--accent-teal)', fontFamily: 'var(--font-mono)' }}>{result.total_score.toFixed(4)}</span>
                            </div>
                        </div>

                        <div style={{ color: 'var(--text-muted)', fontSize: 11, textTransform: 'uppercase', marginBottom: 10 }}>
                            Objective Breakdown
                        </div>

                        {Object.entries(result.factors)
                            .sort(([, a], [, b]) => b - a)
                            .map(([name, score]) => (
                                <div key={name} className="explain-factor">
                                    <span className="factor-name" title={name}>
                                        {name.replace('Objective', '').replace(/([A-Z])/g, ' $1').trim()}
                                    </span>
                                    <div className="factor-bar-bg">
                                        <div
                                            className="factor-bar"
                                            style={{ width: `${Math.min(100, (score / maxScore) * 100)}%` }}
                                        />
                                    </div>
                                    <span className="factor-value">{score.toFixed(4)}</span>
                                </div>
                            ))}

                        <div className="explain-reasoning">
                            {result.reasoning}
                        </div>

                        {result.alternatives.length > 0 && (
                            <div className="explain-alt">
                                <h4>Top Alternatives</h4>
                                {result.alternatives.map((alt, i) => (
                                    <div key={alt.worker_id} className="alt-row">
                                        <span style={{ color: 'var(--text-secondary)' }}>#{i + 2} {alt.worker_id}</span>
                                        <span style={{ color: 'var(--text-muted)' }}>{alt.score.toFixed(4)}</span>
                                    </div>
                                ))}
                            </div>
                        )}
                    </>
                )}
            </div>
        </div>
    )
}
