# MUSA EP 推荐模型验证镜像

镜像：`registry.mthreads.com/presale/devtech/musa_ep_rec:musa438`

镜像内含 MUSA 4.3.8 用户态 runtime、muDNN 3.4.0 recommendation 运行库、JD-MUSA ORT 1.18 动态库、验证脚本、模型及外部权重 `model_data.bin`、800 组输入和 TF32 GT。工作目录为 `/opt/musa_ep_rec`。

宿主机只提供 MUSA driver；toolkit、muDNN、ORT 库、脚本和模型均来自镜像。每台机器需要将与宿主机匹配的 `libmusa.so.*` 只读挂载到容器内的 `/opt/musa-host-driver/libmusa.so.1`。

## 1. 创建容器

将 `<driver-so>` 替换为宿主机实际 driver 文件，将 `<gpu-id>` 替换为要使用的物理卡。示例：

```bash
docker run -d --name musa-ep-rec-musa438 \
  --runtime=mthreads --privileged --network=host --shm-size=80g \
  -e MTHREADS_VISIBLE_DEVICES=all \
  -e MTHREADS_DRIVER_CAPABILITIES=all \
  -e MUSA_VISIBLE_DEVICES=<gpu-id> \
  -v <driver-so>:/opt/musa-host-driver/libmusa.so.1:ro \
  registry.mthreads.com/presale/devtech/musa_ep_rec:musa438 sleep infinity

docker exec -it musa-ep-rec-musa438 bash
```

例如， 使用物理 GPU 2 和宿主机 `/usr/lib/x86_64-linux-gnu/libmusa.so.4.3.8`。实际交付时以目标主机的 driver 文件为准。由于：MUSA_VISIBLE_DEVICES=<gpu-id>  会把宿主机物理 GPU <gpu-id>  映射为容器内逻辑 GPU 0，因此测试脚本中仍应使用：

  --device-id 0

## 2. 初始化环境并校验

以下变量必须在执行验证命令的同一个 shell 中设置。镜像已通过 `ENV` 预设这些值，重新显式设置可以避免宿主机或启动脚本覆盖。

```bash
cd /opt/musa_ep_rec

# 设备和路径
export MUSA_VISIBLE_DEVICES=${MUSA_VISIBLE_DEVICES:-0}
export MUSA_HOME=/usr/local/musa-4.3.8
export MUSA_EP_REC_ROOT=/opt/musa_ep_rec
export ONNXRUNTIME_ROOT=/opt/musa_ep_rec
export MODEL_DATA_ROOT=/opt/musa_ep_rec/model
export RESULT_ROOT=/opt/musa_ep_rec/results
export ORT_MUSA_ORT_LIB=/opt/musa_ep_rec/runtime/libonnxruntime.so.1.18.0

# 推理配置
export ORT_MUSA_ENABLE_TF32=1
export ORT_MUSA_ALLOCATOR_CACHE_LIMIT_MB=4096
export ORT_SESSION_MEMCPY_USE_MEMORY_POOL_ON=0
export PYTHONUNBUFFERED=1

# 让动态链接器先找到挂载的 driver、镜像内三方库和 MUSA runtime
export LD_LIBRARY_PATH=/opt/musa-host-driver:/opt/musa_ep_rec/thirdparty/METIS_LIB/lib:/opt/musa_ep_rec/thirdparty/gfortran:/usr/local/musa-4.3.8/lib:/usr/local/musa/lib:/usr/lib64:/usr/lib

mkdir -p "$RESULT_ROOT"
sha256sum -c SHA256SUMS
ldd "$ORT_MUSA_ORT_LIB" | grep -E 'musa|mudnn|mublas|metis|gfortran|quadmath'
python3 --version
```

`MODEL_DATA_ROOT` 是模型、`model_data.bin`、输入和 GT 的根目录；`RESULT_ROOT` 是所有验证输出的根目录；`ORT_MUSA_ORT_LIB` 是待验证的 ORT 动态库；`MUSA_HOME` 和 `LD_LIBRARY_PATH` 决定用户态 MUSA/muDNN 库；`ORT_MUSA_ENABLE_TF32` 开启 TF32；`ORT_MUSA_ALLOCATOR_CACHE_LIMIT_MB` 为 allocator cache 上限（4096 MB）；`ORT_SESSION_MEMCPY_USE_MEMORY_POOL_ON` 控制 ORT session memcpy memory pool，当前验证口径为 0。

## 3. 选择同 NUMA 的空闲 CPU

性能、精度和一致性命令都应使用同一组空闲 CPU。先查看 CPU 与 NUMA 的对应关系，再把 `CPU_LIST` 改为目标 NUMA 上的 8 个空闲逻辑 CPU：

```bash
lscpu -e=CPU,NODE,ONLINE
numactl --hardware
export CPU_LIST=0-7       # 示例；交付机器上按实际空闲 NUMA 修改
```

## 4. 性能测试

固定口径：ORT parallel、`parallel-inter-op=8`、CPU arena 关闭、16 workers、uniform 100 QPS、预热 20 秒、采样 60 秒。`latency_p99_ms` 是 `OrtApi::Run` P99；`musa_completion_p99_ms` 是 MUSA 边界完成 P99，不能用 wall time 代替。

```bash
rm -rf "$RESULT_ROOT/parallel"
taskset -c "$CPU_LIST" python3 -u tools/pressure_native_capi.py \
  --model "$MODEL_DATA_ROOT/optimized_model.onnx" \
  --request-dir "$MODEL_DATA_ROOT/req_799_npz" \
  --device-placement-file "$MODEL_DATA_ROOT/device.txt" \
  --ort-library "$ORT_MUSA_ORT_LIB" \
  --device-id 0 --mode uniform --qps 100 --workers 16 \
  --execution-mode parallel --parallel-inter-op 8 \
  --disable-cpu-mem-arena --warmup-seconds 20 --duration 60 \
  --output-dir "$RESULT_ROOT/parallel"

python3 - <<'PY'
import json, os
p = os.path.join(os.environ['RESULT_ROOT'], 'parallel', 'summary.json')
s = json.load(open(p))
for k in ('actual_qps', 'musa_completion_p99_ms', 'latency_p99_ms', 'queue_wait_p99_ms', 'runtime_errors'):
    print(f'{k}: {s.get(k)}')
PY
```

`pressure_native_capi.py` 会在启动时自动设置并在采样窗口开启 boundary timing：

```text
ORT_MUSA_PROFILE_TIMING=1
ORT_MUSA_PROFILE_TIMING_BACKEND=boundary
ORT_MUSA_PROFILE_TIMING_CAPTURE=0 -> 采样窗口自动切为 1 -> 结束后恢复 0
ORT_MUSA_PROFILE_TIMING_SYNC=0
ORT_MUSA_PROFILE_TIMING_OUTPUT=$RESULT_ROOT/parallel/musa_timing.csv
```

因此不需要手工改脚本；如果需要核查，查看 `summary.json` 的 `runtime_environment`。正常输出还应包含 `musa_timing.csv.boundary.csv`、`musa_timing.csv.d2h_sync.csv`、`musa_completion.csv` 和 `diagnostics.csv`。

## 5. 精度测试

```bash
rm -rf "$RESULT_ROOT/precision"
taskset -c "$CPU_LIST" python3 -u tools/run_precision.py \
  --model "$MODEL_DATA_ROOT/optimized_model.onnx" \
  --placement "$MODEL_DATA_ROOT/device.txt" \
  --request-dir "$MODEL_DATA_ROOT/req_799_npz" \
  --gt-dir "$MODEL_DATA_ROOT/req_799_tf32_on_GT" \
  --output-dir "$RESULT_ROOT/precision" \
  --ort-library "$ORT_MUSA_ORT_LIB" --device-id 0 \
  --execution-mode parallel --parallel-inter-op 8 --disable-cpu-mem-arena
```

验收条件：800 个样本、38 个输出、22 个严格指标无超阈值项，最大绝对误差小于 `1e-3`。结果和失败样本索引在 `$RESULT_ROOT/precision`。

## 6. 并发一致性测试

```bash
rm -f "$RESULT_ROOT/consistency.json"
taskset -c "$CPU_LIST" python3 -u tools/run_concurrency_consistency.py \
  --model "$MODEL_DATA_ROOT/optimized_model.onnx" \
  --placement "$MODEL_DATA_ROOT/device.txt" \
  --request-dir "$MODEL_DATA_ROOT/req_799_npz" \
  --output "$RESULT_ROOT/consistency.json" \
  --ort-library "$ORT_MUSA_ORT_LIB" --device-id 0 \
  --execution-mode parallel --parallel-inter-op 8 --disable-cpu-mem-arena \
  --workers 16 --rounds 20
```

验收条件：`all_bitwise_identical=true`、`all_within_1e-3=true`，bitwise/SHA256 mismatch 和超 `1e-3` 计数均为 0。

## 7. 结果回传

至少回传以下文件，便于复核环境和性能：

```text
$RESULT_ROOT/parallel/summary.json
$RESULT_ROOT/parallel/latency.csv
$RESULT_ROOT/parallel/musa_timing.csv*
$RESULT_ROOT/parallel/diagnostics.csv
$RESULT_ROOT/precision/*
$RESULT_ROOT/consistency.json
```
