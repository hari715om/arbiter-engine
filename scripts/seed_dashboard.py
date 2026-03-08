"""One-Time API Seeder — quickly populates the dashboard with realistic data.

This script hits the API once to:
  1. Register 5 workers
  2. Create 30 tasks
  3. Schedule the tasks
  4. Optionally mark some tasks as completed

Usage:
  python scripts/seed_dashboard.py
"""

import time
import httpx
import random

API = "http://localhost:8000"

def main():
    try:
        client = httpx.Client(base_url=API, timeout=5.0)
        # Check API
        client.get("/health")
    except Exception:
        print("ERROR: Cannot reach API. Is docker-compose up running?")
        return

    print("1/4 Registering workers...")
    workers = [
        {"id": "gpu-01", "cpu_capacity": 32, "memory_capacity": 64, "supported_resources": ["cpu", "gpu"]},
        {"id": "gpu-02", "cpu_capacity": 24, "memory_capacity": 48, "supported_resources": ["cpu", "gpu"]},
        {"id": "cpu-01", "cpu_capacity": 16, "memory_capacity": 32, "supported_resources": ["cpu", "memory"]},
        {"id": "cpu-02", "cpu_capacity": 16, "memory_capacity": 32, "supported_resources": ["cpu", "memory"]},
        {"id": "cpu-03", "cpu_capacity": 8,  "memory_capacity": 16, "supported_resources": ["cpu", "memory"]},
    ]
    for w in workers:
        try:
            client.post("/workers", json=w)
            client.post(f"/workers/{w['id']}/heartbeat")
        except:
            pass

    print("2/4 Creating tasks...")
    for i in range(1, 41):
        task = {
            "id": f"task-abc-{i:03d}",
            "compute_cost": random.choice([1.0, 2.0, 3.0, 5.0]),
            "resource_type": "gpu" if i % 5 == 0 else "cpu",
            "deadline": time.time() + 300,
            "priority": random.randint(1, 10),
            "estimated_duration": 5.0,
            "dependencies": [],
        }
        try:
            client.post("/tasks", json=task)
        except:
            pass

    print("3/4 Triggering scheduler...")
    client.post("/schedule")
    
    print("4/4 Marking some tasks as completed...")
    running = client.get("/tasks?status=running").json()
    completed = 0
    for t in running:
        if random.random() < 0.3:
            # We delete the task to mimic completion and free worker capacity
            client.delete(f"/tasks/{t['id']}")
            completed += 1
            
    print(f"\n✅ Dashboard Seeded!")
    print(f"  - 5 Workers active")
    print(f"  - 40 Tasks submitted")
    print(f"  - {completed} Tasks completed")
    print("\nView at: http://localhost:5173")

if __name__ == "__main__":
    main()
