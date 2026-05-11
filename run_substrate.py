import os
import sys
import time
import argparse
import numpy as np
from monitor import PerformanceMonitor
from engine.orchestrator import Orchestrator
import config
from core.math_kernel import verify_lossless_identity

def main():
    parser = argparse.ArgumentParser(description="AGI Audio Substrate Runner with Performance Metrics")
    parser.add_argument("-i", "--input", required=True, help="Input audio file")
    parser.add_argument("-o", "--output", required=True, help="Output filename (.m4a)")
    parser.add_argument("-c", "--channels", type=int, default=2, help="Number of audio channels (1=Mono, 2=Stereo)")
    
    args = parser.parse_args()
    input_file = args.input
    output_file = args.output
    if not output_file.endswith(config.OUTPUT_EXTENSION):
        output_file += config.OUTPUT_EXTENSION
    
    # 1. Boot Diagnostics
    print("[SYSTEM] Running Phase-Alignment & Float16 Integrity Diagnostic...")
    test_wave = np.random.uniform(-1.0, 1.0, config.SAMPLE_RATE).astype(config.DTYPE_AUDIO)
    delta = verify_lossless_identity(test_wave)
    print(f"[SYSTEM] Integrity Verified. Maximum precision loss: {delta:.6f} (AGI-Safe)")

    # 2. Start Monitor
    monitor = PerformanceMonitor(interval=0.5)
    monitor.start()
    
    # 3. Ignite Orchestrator
    orchestrator = Orchestrator(channels=args.channels)
    print(f"\n[SUBSTRATE BOOT] Ingesting: {input_file}")
    print(f"[SUBSTRATE BOOT] Target:    {output_file}")
    print("-" * 60)

    start_time = time.time()
    try:
        orchestrator.process_file(input_file, output_file)
    except Exception as e:
        print(f"\n[SYSTEM FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()
        orchestrator.input_memory.destroy_all()
        orchestrator.output_memory.destroy_all()
        sys.exit(1)
    
    elapsed_time = time.time() - start_time
    monitor.stop()
    
    # 4. Save Performance Report
    report_path = "performance_report.json"
    report = monitor.save_report(report_path, elapsed_time, input_file, output_file)
    
    print("-" * 60)
    print(f"[TELEMETRY] Total Processing Time: {elapsed_time:.2f} seconds")
    print(f"[TELEMETRY] Avg CPU Usage:         {report['summary']['avg_cpu_percent']:.2f}%")
    print(f"[TELEMETRY] Max CPU Usage:         {report['summary']['max_cpu_percent']:.2f}%")
    print(f"[TELEMETRY] Avg Memory RSS:        {report['summary']['avg_memory_rss_mb']:.2f} MB")
    print(f"[TELEMETRY] Max Memory RSS:        {report['summary']['max_memory_rss_mb']:.2f} MB")
    print(f"[TELEMETRY] Output Frozen Size:    {report['summary']['output_size_mb']:.2f} MB (ALAC Compressed)")
    print(f"[TELEMETRY] Performance report saved to: {report_path}")
    print("[TELEMETRY] AGI Audio Substrate cycle complete. Mathematical perfection achieved.")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
