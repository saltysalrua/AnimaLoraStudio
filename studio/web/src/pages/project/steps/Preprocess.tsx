import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link, useOutletContext } from 'react-router-dom'
import {
  api,
  type Job,
  type PreprocessedItem,
  type PreprocessPendingItem,
  type ProjectDetail,
  type UpscalerVariant,
  type Version,
} from '../../../api/client'
import ImageGrid, { applySelection } from '../../../components/ImageGrid'
import StepShell from '../../../components/StepShell'
import { useDialog } from '../../../components/Dialog'
import { useToast } from '../../../components/Toast'
import { useEventStream } from '../../../lib/useEventStream'

interface Ctx {
  project: ProjectDetail
  activeVersion: Version | null
  reload: () => Promise<void>
}

interface Status {
  job: Job | null
  log_tail: string
  summary: { download_count: number; processed_count: number; pending_count: number }
}

interface FilesView {
  processed: PreprocessedItem[]
  pending: PreprocessPendingItem[]
  summary: Status['summary']
}

const STATUS_COLOR: Record<Job['status'], string> = {
  pending: 'badge badge-neutral',
  running: 'badge badge-warn',
  done: 'badge badge-ok',
  failed: 'badge badge-err',
  canceled: 'badge badge-neutral',
}

const DEFAULT_MODEL = '4x-AnimeSharp'
const TILE_OPTIONS = [128, 192, 256, 384, 512] as const
type Device = 'auto' | 'cuda' | 'cpu'
const DEVICE_OPTIONS: { value: Device; label: string }[] = [
  { value: 'auto', label: '自动（优先 CUDA）' },
  { value: 'cuda', label: 'CUDA' },
  { value: 'cpu', label: 'CPU' },
]

export default function PreprocessPage() {
  const { project, reload } = useOutletContext<Ctx>()
  const { toast } = useToast()
  const { confirm } = useDialog()

  const [files, setFiles] = useState<FilesView | null>(null)
  const [status, setStatus] = useState<Status | null>(null)
  const [logs, setLogs] = useState<string[]>([])
  const [busy, setBusy] = useState(false)
  const [tileSize, setTileSize] = useState<number>(256)
  const [device, setDevice] = useState<Device>('auto')
  const [pendingSel, setPendingSel] = useState<Set<string>>(new Set())
  const [pendingAnchor, setPendingAnchor] = useState<string | null>(null)
  const [processedSel, setProcessedSel] = useState<Set<string>>(new Set())
  const [processedAnchor, setProcessedAnchor] = useState<string | null>(null)

  // 模型权重就绪状态（catalog 取一次，下载完成后用户手动刷新或 SSE 更新）
  const [upscaler, setUpscaler] = useState<UpscalerVariant | null>(null)
  const [downloadingModel, setDownloadingModel] = useState(false)

  const refreshFiles = useCallback(async () => {
    try {
      const r = await api.listPreprocessFiles(project.id)
      setFiles(r)
    } catch {
      /* ignore */
    }
  }, [project.id])

  const refreshStatus = useCallback(async () => {
    try {
      const r = await api.getPreprocessStatus(project.id)
      setStatus(r)
      setLogs(r.log_tail ? r.log_tail.split('\n') : [])
    } catch {
      /* ignore */
    }
  }, [project.id])

  const refreshUpscaler = useCallback(async () => {
    try {
      const cat = await api.getModelsCatalog()
      const v = cat.upscalers?.variants?.find((x) => x.label === DEFAULT_MODEL) ?? null
      setUpscaler(v)
    } catch {
      /* ignore */
    }
  }, [])

  useEffect(() => {
    void refreshFiles()
    void refreshStatus()
    void refreshUpscaler()
  }, [refreshFiles, refreshStatus, refreshUpscaler])

  // SSE：job 状态变化 → 刷 status + files；model download 完成 → 刷 catalog
  const jobIdRef = useRef<number | null>(null)
  jobIdRef.current = status?.job?.id ?? null
  useEventStream((evt) => {
    const jid = jobIdRef.current
    if (evt.type === 'job_log_appended' && jid && evt.job_id === jid) {
      setLogs((prev) => [...prev, String(evt.text ?? '')])
    } else if (evt.type === 'job_state_changed' && jid && evt.job_id === jid) {
      void refreshStatus()
      if (evt.status === 'done' || evt.status === 'failed' || evt.status === 'canceled') {
        void refreshFiles()
        void reload()
      }
    } else if (evt.type === 'project_state_changed' && evt.project_id === project.id) {
      void refreshFiles()
    } else if (evt.type === 'model_download_changed') {
      void refreshUpscaler()
    }
  })

  const job = status?.job ?? null
  const isLive = job?.status === 'running' || job?.status === 'pending'
  const summary = files?.summary ?? status?.summary ?? {
    download_count: 0,
    processed_count: 0,
    pending_count: 0,
  }
  const modelReady = !!upscaler?.exists

  // ----- 操作 ---------------------------------------------------------------
  const downloadModel = async () => {
    if (downloadingModel) return
    setDownloadingModel(true)
    try {
      await api.startModelDownload({ model_id: 'upscaler', variant: DEFAULT_MODEL })
      toast('开始下载 4x-AnimeSharp', 'success')
      // 让 SSE 推 model_download_changed；这里也立即刷一下兜底
      setTimeout(() => void refreshUpscaler(), 1500)
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setDownloadingModel(false)
    }
  }

  const startPreprocess = async (
    mode: 'all' | 'selected' | 'all_force',
    names?: string[],
  ) => {
    if (!modelReady) {
      toast('请先下载 4x-AnimeSharp 模型', 'error')
      return
    }
    setBusy(true)
    try {
      const j = await api.startPreprocess(project.id, {
        mode,
        names,
        model: DEFAULT_MODEL,
        tile_size: tileSize,
        device,
      })
      setLogs([])
      setStatus((prev) => ({
        job: j,
        log_tail: '',
        summary: prev?.summary ?? summary,
      }))
      toast(`开始预处理 #${j.id}`, 'success')
      setPendingSel(new Set())
      setPendingAnchor(null)
      setProcessedSel(new Set())
      setProcessedAnchor(null)
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setBusy(false)
    }
  }

  const cancel = async () => {
    if (!job) return
    try {
      await api.cancelJob(job.id)
      toast('已取消', 'success')
    } catch (e) {
      toast(String(e), 'error')
    }
  }

  const deleteProcessed = async () => {
    if (processedSel.size === 0) return
    if (!(await confirm(
      `从 preprocess/ 删除 ${processedSel.size} 张产物（这些图会回到「待处理」列表）？`,
      { tone: 'danger', okText: '删除' },
    ))) return
    try {
      const r = await api.deletePreprocessFiles(project.id, Array.from(processedSel))
      toast(
        `已删除 ${r.deleted.length} 张${r.missing.length ? ` · 跳过 ${r.missing.length}` : ''}`,
        'success',
      )
      setProcessedSel(new Set())
      setProcessedAnchor(null)
      await refreshFiles()
      void reload()
    } catch (e) {
      toast(String(e), 'error')
    }
  }

  // ----- grids --------------------------------------------------------------
  const pendingNames = useMemo(
    () => (files?.pending ?? []).map((it) => it.name),
    [files],
  )
  const processedNames = useMemo(
    () => (files?.processed ?? []).map((it) => it.name),
    [files],
  )

  const pendingItems = useMemo(
    () =>
      pendingNames.map((n) => ({
        name: n,
        thumbUrl: api.projectThumbUrl(project.id, n),
      })),
    [pendingNames, project.id],
  )
  const processedItems = useMemo(
    () =>
      (files?.processed ?? []).map((it) => ({
        name: it.name,
        thumbUrl: api.preprocessThumbUrl(project.id, it.name),
        meta: it.dst_size
          ? `${it.src_size?.join('×') ?? '?'} → ${it.dst_size.join('×')} (${it.scale}×)`
          : undefined,
      })),
    [files, project.id],
  )

  return (
    <StepShell
      idx={2}
      title="预处理"
      subtitle="放大 / 裁剪 / 涂抹 — 第一阶段：放大"
      actions={
        <Link to="/tools/settings" className="btn btn-ghost btn-sm">
          设置
        </Link>
      }
    >
      <div className="flex flex-col h-full gap-3 min-h-0">
        <div className="grid gap-3 flex-1 min-h-0" style={{ gridTemplateColumns: '1fr 260px' }}>
          {/* 左栏 */}
          <div className="flex flex-col gap-2 min-h-0 min-w-0">
            <OperationPanel
              tileSize={tileSize}
              setTileSize={setTileSize}
              device={device}
              setDevice={setDevice}
              modelReady={modelReady}
              downloadingModel={downloadingModel}
              onDownloadModel={() => void downloadModel()}
              upscaler={upscaler}
              pendingCount={summary.pending_count}
              pendingSelCount={pendingSel.size}
              busy={busy || isLive}
              onStartAll={() => void startPreprocess('all')}
              onStartSelected={() =>
                void startPreprocess('selected', Array.from(pendingSel))
              }
            />

            {job && (
              <JobStrip
                job={job}
                logs={logs}
                onCancel={isLive ? cancel : undefined}
              />
            )}

            {/* 已处理 grid */}
            <ProcessedSection
              items={processedItems}
              selected={processedSel}
              onSelect={(name, e) => {
                const r = applySelection(processedSel, name, e, processedNames, processedAnchor)
                setProcessedSel(r.next)
                setProcessedAnchor(r.anchor)
              }}
              onSelectAll={() => setProcessedSel(new Set(processedNames))}
              onClear={() => {
                setProcessedSel(new Set())
                setProcessedAnchor(null)
              }}
              onDelete={() => void deleteProcessed()}
            />

            {/* 待处理 grid */}
            <PendingSection
              items={pendingItems}
              selected={pendingSel}
              onSelect={(name, e) => {
                const r = applySelection(pendingSel, name, e, pendingNames, pendingAnchor)
                setPendingSel(r.next)
                setPendingAnchor(r.anchor)
              }}
              onSelectAll={() => setPendingSel(new Set(pendingNames))}
              onClear={() => {
                setPendingSel(new Set())
                setPendingAnchor(null)
              }}
            />
          </div>

          {/* 右栏统计 */}
          <PreprocessSidebar
            summary={summary}
            upscaler={upscaler}
            tileSize={tileSize}
          />
        </div>
      </div>
    </StepShell>
  )
}

// ---------------------------------------------------------------------------
// 操作 panel
// ---------------------------------------------------------------------------

interface OperationPanelProps {
  tileSize: number
  setTileSize: (n: number) => void
  device: Device
  setDevice: (d: Device) => void
  modelReady: boolean
  downloadingModel: boolean
  onDownloadModel: () => void
  upscaler: UpscalerVariant | null
  pendingCount: number
  pendingSelCount: number
  busy: boolean
  onStartAll: () => void
  onStartSelected: () => void
}

function OperationPanel({
  tileSize,
  setTileSize,
  device,
  setDevice,
  modelReady,
  downloadingModel,
  onDownloadModel,
  upscaler,
  pendingCount,
  pendingSelCount,
  busy,
  onStartAll,
  onStartSelected,
}: OperationPanelProps) {
  return (
    <section className="flex flex-col gap-1.5 rounded-md border border-subtle bg-surface px-3 py-2.5 shrink-0">
      <h3 className="caption flex items-center gap-1.5">
        <span className="inline-block w-1.5 h-1.5 rounded-full shrink-0 bg-accent" />
        放大设置
      </h3>

      {!modelReady && (
        <div className="flex items-center gap-2 text-sm px-2 py-1.5 rounded-sm bg-warn-soft border border-warn">
          <span className="text-warn font-medium">需要下载模型</span>
          <span className="text-fg-secondary text-xs flex-1 truncate">
            {upscaler?.repo ?? 'Kim2091/AnimeSharp'} · ~64 MB
          </span>
          <button
            onClick={onDownloadModel}
            disabled={downloadingModel}
            className="btn btn-primary btn-sm"
          >
            {downloadingModel ? '下载中...' : '下载 4x-AnimeSharp'}
          </button>
        </div>
      )}

      <div className="flex items-center gap-2 text-sm flex-wrap">
        <label className="flex items-center gap-1.5">
          <span className="text-fg-tertiary">模型</span>
          <span className="mono text-fg-primary">{DEFAULT_MODEL}</span>
          <span className="text-fg-tertiary">· scale=4</span>
        </label>

        <span className="text-dim">·</span>

        <label className="flex items-center gap-1.5">
          <span className="text-fg-tertiary">tile</span>
          <select
            value={tileSize}
            onChange={(e) => setTileSize(Number(e.target.value))}
            disabled={busy}
            className="input text-sm"
            style={{ width: 'auto', padding: '2px 6px' }}
          >
            {TILE_OPTIONS.map((n) => (
              <option key={n} value={n}>{n}px</option>
            ))}
          </select>
        </label>

        <span className="text-dim">·</span>

        <label className="flex items-center gap-1.5">
          <span className="text-fg-tertiary">设备</span>
          <select
            value={device}
            onChange={(e) => setDevice(e.target.value as Device)}
            disabled={busy}
            className="input text-sm"
            style={{ width: 'auto', padding: '2px 6px' }}
          >
            {DEVICE_OPTIONS.map((o) => (
              <option key={o.value} value={o.value}>{o.label}</option>
            ))}
          </select>
        </label>

        <span className="flex-1" />

        <button
          onClick={onStartSelected}
          disabled={busy || !modelReady || pendingSelCount === 0}
          className="btn btn-secondary btn-sm"
          title={pendingSelCount === 0 ? '在「待处理」里选图后启用' : ''}
        >
          {`放大选中 ${pendingSelCount}`}
        </button>
        <button
          onClick={onStartAll}
          disabled={busy || !modelReady || pendingCount === 0}
          className="btn btn-primary btn-sm"
        >
          {pendingCount > 0 ? `放大全部 ${pendingCount}` : '没有待处理'}
        </button>
      </div>

      {/* 未来 tabs 占位，提示用户裁剪 / 涂抹 还没上线 */}
      <div className="flex items-center gap-2 mt-1 text-xs text-fg-tertiary">
        <span className="font-medium text-fg-secondary">阶段</span>
        <span className="px-1.5 py-0.5 rounded bg-accent-soft text-accent text-xs font-medium">放大</span>
        <span
          className="px-1.5 py-0.5 rounded bg-overlay opacity-50 cursor-not-allowed"
          title="未来阶段：交互式裁剪"
        >裁剪</span>
        <span
          className="px-1.5 py-0.5 rounded bg-overlay opacity-50 cursor-not-allowed"
          title="未来阶段：画笔涂抹（取色 + 高斯）"
        >涂抹</span>
      </div>
    </section>
  )
}

// ---------------------------------------------------------------------------
// 已处理 / 待处理 sections
// ---------------------------------------------------------------------------

function ProcessedSection({
  items,
  selected,
  onSelect,
  onSelectAll,
  onClear,
  onDelete,
}: {
  items: { name: string; thumbUrl: string; meta?: string }[]
  selected: Set<string>
  onSelect: (name: string, e: React.MouseEvent) => void
  onSelectAll: () => void
  onClear: () => void
  onDelete: () => void
}) {
  return (
    <section className="flex flex-col flex-1 min-h-0 rounded-md border border-subtle bg-surface overflow-hidden">
      <header className="flex items-center gap-2 shrink-0 px-2.5 py-1.5 border-b border-subtle text-sm">
        <h3 className="font-semibold">已处理</h3>
        <span className="text-fg-tertiary">{items.length} 张</span>
        {selected.size > 0 && (
          <span className="text-accent">· 已选 {selected.size}</span>
        )}
        <span className="flex-1" />
        <button
          onClick={onSelectAll}
          disabled={items.length === 0}
          className="btn btn-ghost btn-sm"
        >全选</button>
        <button
          onClick={onClear}
          disabled={selected.size === 0}
          className="btn btn-ghost btn-sm"
        >清空</button>
        <button
          onClick={onDelete}
          disabled={selected.size === 0}
          className="btn btn-sm bg-err-soft text-err"
          title="删除选中产物（源会重新出现在「待处理」）"
        >🗑 删除 {selected.size}</button>
      </header>
      <div className="flex-1 min-h-0 overflow-y-auto p-2">
        <ImageGrid
          items={items}
          selected={selected}
          onSelect={onSelect}
          ariaLabel="preprocess-processed-grid"
          emptyHint="还没有产物 — 点击上方「放大全部 N」"
        />
      </div>
    </section>
  )
}

function PendingSection({
  items,
  selected,
  onSelect,
  onSelectAll,
  onClear,
}: {
  items: { name: string; thumbUrl: string }[]
  selected: Set<string>
  onSelect: (name: string, e: React.MouseEvent) => void
  onSelectAll: () => void
  onClear: () => void
}) {
  return (
    <section className="flex flex-col flex-1 min-h-0 rounded-md border border-subtle bg-surface overflow-hidden">
      <header className="flex items-center gap-2 shrink-0 px-2.5 py-1.5 border-b border-subtle text-sm">
        <h3 className="font-semibold">待处理</h3>
        <span className="text-fg-tertiary">{items.length} 张</span>
        {selected.size > 0 && (
          <span className="text-accent">· 已选 {selected.size}</span>
        )}
        <span className="flex-1" />
        <button
          onClick={onSelectAll}
          disabled={items.length === 0}
          className="btn btn-ghost btn-sm"
        >全选</button>
        <button
          onClick={onClear}
          disabled={selected.size === 0}
          className="btn btn-ghost btn-sm"
        >清空</button>
      </header>
      <div className="flex-1 min-h-0 overflow-y-auto p-2">
        <ImageGrid
          items={items}
          selected={selected}
          onSelect={onSelect}
          ariaLabel="preprocess-pending-grid"
          emptyHint="所有图都已预处理 ✓"
        />
      </div>
    </section>
  )
}

// ---------------------------------------------------------------------------
// JobStrip
// ---------------------------------------------------------------------------

function JobStrip({
  job,
  logs,
  onCancel,
}: {
  job: Job
  logs: string[]
  onCancel?: () => void
}) {
  const elapsed =
    job.started_at && (job.finished_at ?? Date.now() / 1000) - job.started_at
  const isLive = job.status === 'running' || job.status === 'pending'
  const lastLine = logs[logs.length - 1] ?? ''
  return (
    <details
      open={isLive}
      className="group rounded-md border border-subtle bg-surface overflow-hidden shrink-0"
    >
      <summary className="cursor-pointer flex items-center gap-2 list-none px-2.5 py-1.5 text-sm select-none">
        <span className="inline-block transition-transform group-open:rotate-90 text-fg-tertiary w-3">▸</span>
        <span className={STATUS_COLOR[job.status]}>{job.status}</span>
        <span className="mono text-fg-secondary">job #{job.id}</span>
        {elapsed && elapsed > 0 && (
          <span className="text-fg-tertiary">· {Math.round(elapsed)}s</span>
        )}
        <span className="mono truncate flex-1 min-w-0 text-fg-secondary text-xs">
          {lastLine}
        </span>
        {isLive && onCancel && (
          <button
            onClick={(e) => {
              e.preventDefault()
              onCancel()
            }}
            className="btn btn-ghost btn-sm text-err"
          >取消</button>
        )}
      </summary>
      <pre className="px-3 py-2 text-xs font-mono text-fg-secondary bg-sunken max-h-[224px] overflow-auto whitespace-pre-wrap border-t border-subtle m-0">
        {logs.length === 0 ? '(等待日志...)' : logs.slice(-1000).join('\n')}
      </pre>
    </details>
  )
}

// ---------------------------------------------------------------------------
// 右栏侧边栏
// ---------------------------------------------------------------------------

function PreprocessSidebar({
  summary,
  upscaler,
  tileSize,
}: {
  summary: Status['summary']
  upscaler: UpscalerVariant | null
  tileSize: number
}) {
  const { download_count, processed_count, pending_count } = summary
  const pct = download_count > 0 ? Math.round((processed_count / download_count) * 100) : 0
  // 粗略 VRAM 估算：tile²×scale²×4byte×7倍中间张量，单位 MB。仅给用户一个量级。
  const estVramMB = Math.round((tileSize * tileSize * 16 * 4 * 7) / (1024 * 1024))

  return (
    <div className="flex flex-col gap-3 min-w-0">
      <div className="rounded-md border border-subtle bg-surface px-3 py-2.5">
        <h3 className="caption flex items-center gap-1.5">
          <span className="inline-block w-1.5 h-1.5 rounded-full shrink-0 bg-accent" />
          预处理进度
        </h3>
        <StatRow label="源 download/" value={`${download_count} 张`} />
        <StatRow label="已处理" value={`${processed_count} 张`} accent="ok" />
        <StatRow label="待处理" value={`${pending_count} 张`} accent={pending_count > 0 ? 'warn' : undefined} />
        <div className="mt-2 h-1.5 rounded bg-sunken overflow-hidden">
          <div
            className="h-full bg-accent rounded transition-[width] duration-300 ease-out"
            style={{ width: `${pct}%` }}
          />
        </div>
        <p className="text-xs text-fg-tertiary mt-1 text-right">{pct}%</p>
      </div>

      <div className="rounded-md border border-subtle bg-surface px-3 py-2.5">
        <h3 className="caption flex items-center gap-1.5">
          <span className="inline-block w-1.5 h-1.5 rounded-full shrink-0 bg-ok" />
          模型
        </h3>
        <StatRow
          label={DEFAULT_MODEL}
          value={upscaler?.exists ? '已就绪' : '未下载'}
          accent={upscaler?.exists ? 'ok' : 'warn'}
        />
        {upscaler?.exists && (
          <StatRow
            label="大小"
            value={`${(upscaler.size / 1024 / 1024).toFixed(1)} MB`}
          />
        )}
        <StatRow label="估算 VRAM 峰值" value={`~${estVramMB} MB`} />
        <p className="text-[11px] text-fg-tertiary mt-1.5 leading-snug">
          tile 越大显存占用越高；显存吃紧时降到 128/192。
        </p>
      </div>

      <div className="rounded-md border border-subtle bg-surface px-3 py-2.5">
        <h3 className="caption flex items-center gap-1.5">
          <span className="inline-block w-1.5 h-1.5 rounded-full shrink-0 bg-warn opacity-60" />
          下一步
        </h3>
        <p className="text-xs text-fg-secondary leading-snug">
          预处理产物存在时，<strong className="text-accent">筛选</strong>页左侧自动切到 preprocess/。
          预处理是可选阶段 — 跳过也能直接进筛选。
        </p>
      </div>
    </div>
  )
}

function StatRow({
  label,
  value,
  accent,
}: {
  label: string
  value: string | number
  accent?: 'ok' | 'warn' | 'err'
}) {
  const cls =
    accent === 'ok' ? 'text-ok' :
    accent === 'warn' ? 'text-warn' :
    accent === 'err' ? 'text-err' :
    'text-fg-primary'
  return (
    <div className="flex justify-between items-baseline mt-1.5 text-xs">
      <span className="text-fg-tertiary">{label}</span>
      <span className={`font-mono font-medium ${cls}`}>{value}</span>
    </div>
  )
}
