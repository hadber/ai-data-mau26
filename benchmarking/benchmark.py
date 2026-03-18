from ultralytics.utils.benchmarks import benchmark

# Benchmark on GPU
benchmark(model="best.pt", data="custom.yaml", imgsz=640, half=False, device=0)