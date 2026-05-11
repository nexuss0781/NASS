import time
import psutil
import os
import threading
import numpy as np
import json

class PerformanceMonitor:
    def __init__(self, interval=0.1):
        self.interval = interval
        self.stop_event = threading.Event()
        self.metrics = {
            "timestamp": [],
            "cpu_percent": [],
            "memory_rss_mb": [],
            "memory_vms_mb": [],
            "num_threads": []
        }
        self.process = psutil.Process(os.getpid())

    def _monitor(self):
        while not self.stop_event.is_set():
            try:
                with self.process.oneshot():
                    self.metrics["timestamp"].append(time.time())
                    self.metrics["cpu_percent"].append(self.process.cpu_percent())
                    mem = self.process.memory_info()
                    self.metrics["memory_rss_mb"].append(mem.rss / (1024 * 1024))
                    self.metrics["memory_vms_mb"].append(mem.vms / (1024 * 1024))
                    self.metrics["num_threads"].append(self.process.num_threads())
                
                # Also track child processes (workers)
                children = self.process.children(recursive=True)
                for child in children:
                    try:
                        # We aggregate child CPU/Mem into the main metrics for a total system view
                        self.metrics["cpu_percent"][-1] += child.cpu_percent()
                        self.metrics["memory_rss_mb"][-1] += child.memory_info().rss / (1024 * 1024)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                        
                time.sleep(self.interval)
            except Exception:
                break

    def start(self):
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()
        return self.metrics

    def save_report(self, path, total_time, input_file, output_file):
        report = {
            "summary": {
                "total_processing_time_sec": total_time,
                "avg_cpu_percent": np.mean(self.metrics["cpu_percent"]),
                "max_cpu_percent": np.max(self.metrics["cpu_percent"]),
                "avg_memory_rss_mb": np.mean(self.metrics["memory_rss_mb"]),
                "max_memory_rss_mb": np.max(self.metrics["memory_rss_mb"]),
                "input_file": input_file,
                "output_file": output_file,
                "input_size_mb": os.path.getsize(input_file) / (1024 * 1024),
                "output_size_mb": os.path.getsize(output_file) / (1024 * 1024),
            },
            "raw_metrics": self.metrics
        }
        with open(path, 'w') as f:
            json.dump(report, f, indent=4)
        return report
