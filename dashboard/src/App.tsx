import { useEffect, useRef, useState } from 'react'
import MetricsPanel from './components/MetricsPanel'
import WorkerHeatmap from './components/WorkerHeatmap'
import TaskTimeline from './components/TaskTimeline'
import ExplainPanel from './components/ExplainPanel'

const API = '/api'

export default function App() {
    const [schedulerType, setSchedulerType] = useState('utility')
    const [wsStatus, setWsStatus] = useState<'connecting' | 'live' | 'offline'>('connecting')
    const wsRef = useRef<WebSocket | null>(null)
    const [events, setEvents] = useState<any[]>([])

    // WebSocket connection for live events
    useEffect(() => {
        const connect = () => {
            const ws = new WebSocket('ws://localhost:8000/ws/events')
            wsRef.current = ws
            ws.onopen = () => setWsStatus('live')
            ws.onclose = () => {
                setWsStatus('offline')
                setTimeout(connect, 5000)               // reconnect
            }
            ws.onerror = () => setWsStatus('offline')
            ws.onmessage = (e) => {
                try {
                    const ev = JSON.parse(e.data)
                    setEvents(prev => [ev, ...prev].slice(0, 200))
                } catch { /* ignore */ }
            }
        }
        connect()
        return () => wsRef.current?.close()
    }, [])

    // Fetch scheduler type from health endpoint
    useEffect(() => {
        fetch(`${API}/health`)
            .then(r => r.json())
            .then(d => setSchedulerType(d.scheduler_type || 'utility'))
            .catch(() => { })
    }, [])

    return (
        <div className="app-shell">
            <header className="app-header">
                <h1>⚡ Arbiter Engine</h1>
                <span className={`header-badge badge-live`}>
                    {wsStatus === 'live' ? '● LIVE' : wsStatus === 'offline' ? '○ OFFLINE' : '◌ CONNECTING'}
                </span>
                <span className="header-badge badge-sched">{schedulerType}</span>
                <div className="header-spacer" />
                <div className="header-status">
                    <div className={`status-dot`} style={{ background: wsStatus === 'live' ? 'var(--accent-green)' : 'var(--accent-amber)' }} />
                    <span>WebSocket {wsStatus}</span>
                </div>
            </header>

            <main className="app-grid">
                <MetricsPanel apiBase={API} className="area-metrics card" />
                <WorkerHeatmap apiBase={API} className="area-workers card" />
                <TaskTimeline apiBase={API} events={events} className="area-timeline card" />
                <ExplainPanel apiBase={API} className="area-explain card" />
            </main>
        </div>
    )
}
