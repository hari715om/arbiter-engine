import { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';

/*
 * Feature 5 (Group E): DAG Visualization — force-directed task dependency graph.
 *
 * Polls GET /tasks every 5s. Each task is a node, colored by status:
 *   pending/queued → gray, running → amber, completed → green, failed → red.
 * Dependency edges are rendered as directed arrows.
 * Clicking a node fires onSelectTask → opens the ExplainPanel.
 */

interface Task {
    id: string;
    status: string;
    dependencies: string[];
    priority: number;
    compute_cost: number;
    assigned_worker: string | null;
}

interface DagViewProps {
    onSelectTask?: (taskId: string) => void;
}

const STATUS_COLORS: Record<string, string> = {
    pending: '#6b7280',  // gray
    queued: '#6b7280',
    running: '#f59e0b',  // amber
    completed: '#10b981', // green
    failed: '#ef4444',    // red
};

export default function DagView({ onSelectTask }: DagViewProps) {
    const svgRef = useRef<SVGSVGElement>(null);
    const [tasks, setTasks] = useState<Task[]>([]);
    const [error, setError] = useState<string | null>(null);

    // Poll tasks every 5s
    useEffect(() => {
        const fetchTasks = async () => {
            try {
                const res = await fetch('/api/tasks?limit=200');
                if (res.ok) {
                    const data = await res.json();
                    setTasks(data);
                    setError(null);
                }
            } catch (e) {
                setError('Failed to load tasks');
            }
        };
        fetchTasks();
        const interval = setInterval(fetchTasks, 5000);
        return () => clearInterval(interval);
    }, []);

    // Render D3 force graph
    useEffect(() => {
        if (!svgRef.current || tasks.length === 0) return;

        const svg = d3.select(svgRef.current);
        svg.selectAll('*').remove();

        const width = svgRef.current.clientWidth || 800;
        const height = 400;

        // Build nodes + links from tasks
        const taskIds = new Set(tasks.map(t => t.id));
        const nodes = tasks.map(t => ({
            id: t.id,
            status: t.status,
            priority: t.priority,
        }));

        const links: { source: string; target: string }[] = [];
        tasks.forEach(t => {
            t.dependencies.forEach(dep => {
                if (taskIds.has(dep)) {
                    links.push({ source: dep, target: t.id });
                }
            });
        });

        // Arrow marker
        svg.append('defs').append('marker')
            .attr('id', 'dag-arrow')
            .attr('viewBox', '0 -5 10 10')
            .attr('refX', 18)
            .attr('refY', 0)
            .attr('markerWidth', 6)
            .attr('markerHeight', 6)
            .attr('orient', 'auto')
            .append('path')
            .attr('d', 'M0,-5L10,0L0,5')
            .attr('fill', '#64748b');

        const g = svg.append('g');

        // Zoom
        const zoom = d3.zoom<SVGSVGElement, unknown>()
            .scaleExtent([0.3, 3])
            .on('zoom', (event) => {
                g.attr('transform', event.transform);
            });
        svg.call(zoom);

        // Force simulation
        const simulation = d3.forceSimulation(nodes as any)
            .force('link', d3.forceLink(links as any).id((d: any) => d.id).distance(80))
            .force('charge', d3.forceManyBody().strength(-200))
            .force('center', d3.forceCenter(width / 2, height / 2))
            .force('collide', d3.forceCollide(20));

        // Links
        const link = g.append('g')
            .selectAll('line')
            .data(links)
            .join('line')
            .attr('stroke', '#475569')
            .attr('stroke-width', 1.5)
            .attr('marker-end', 'url(#dag-arrow)');

        // Nodes
        const node = g.append('g')
            .selectAll('circle')
            .data(nodes)
            .join('circle')
            .attr('r', (d: any) => 6 + (d.priority || 5))
            .attr('fill', (d: any) => STATUS_COLORS[d.status] || '#6b7280')
            .attr('stroke', '#1e293b')
            .attr('stroke-width', 1.5)
            .attr('cursor', 'pointer')
            .on('click', (_event: any, d: any) => {
                if (onSelectTask) onSelectTask(d.id);
            });

        // Tooltip on hover
        node.append('title')
            .text((d: any) => `${d.id}\nStatus: ${d.status}\nPriority: ${d.priority}`);

        // Labels
        const label = g.append('g')
            .selectAll('text')
            .data(nodes)
            .join('text')
            .attr('font-size', '9px')
            .attr('fill', '#94a3b8')
            .attr('text-anchor', 'middle')
            .attr('dy', -14)
            .text((d: any) => d.id.length > 12 ? d.id.slice(0, 12) + '…' : d.id);

        // Drag behaviour
        const drag = d3.drag<SVGCircleElement, any>()
            .on('start', (event, d) => {
                if (!event.active) simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
            })
            .on('drag', (event, d) => {
                d.fx = event.x;
                d.fy = event.y;
            })
            .on('end', (event, d) => {
                if (!event.active) simulation.alphaTarget(0);
                d.fx = null;
                d.fy = null;
            });
        node.call(drag as any);

        simulation.on('tick', () => {
            link
                .attr('x1', (d: any) => d.source.x)
                .attr('y1', (d: any) => d.source.y)
                .attr('x2', (d: any) => d.target.x)
                .attr('y2', (d: any) => d.target.y);
            node
                .attr('cx', (d: any) => d.x)
                .attr('cy', (d: any) => d.y);
            label
                .attr('x', (d: any) => d.x)
                .attr('y', (d: any) => d.y);
        });

        return () => { simulation.stop(); };
    }, [tasks, onSelectTask]);

    return (
        <section className="panel dag-panel">
            <h2>Task DAG</h2>
            {error && <p className="error-text">{error}</p>}
            {tasks.length === 0 && !error && (
                <p className="muted-text">No tasks to visualize</p>
            )}
            <div className="dag-legend">
                {Object.entries(STATUS_COLORS).filter(([k]) => !['queued'].includes(k)).map(([status, color]) => (
                    <span key={status} className="legend-item">
                        <span className="legend-dot" style={{ background: color }} />
                        {status}
                    </span>
                ))}
            </div>
            <svg
                ref={svgRef}
                width="100%"
                height={400}
                style={{ background: '#0f172a', borderRadius: '8px' }}
            />
        </section>
    );
}
