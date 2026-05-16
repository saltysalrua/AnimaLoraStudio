import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Link, useNavigate } from 'react-router-dom'
import { api, type Task, type TaskStatus } from '../api/client'
import StepShell from '../components/StepShell'
import { useDialog } from '../components/Dialog'
import { useToast } from '../components/Toast'
import { useEventStream } from '../lib/useEventStream'
import { useMonitorProgress } from '../lib/useMonitorProgress'

async function downloadJson(filename: string, data: unknown) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url; a.download = filename; a.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}

async function pickJsonFile(jsonErrorMsg: string): Promise<unknown | null> {
  return new Promise((resolve, reject) => {
    const input = document.createElement('input')
    input.type = 'file'; input.accept = '.json,application/json'
    input.onchange = async () => {
      const f = input.files?.[0]
      if (!f) { resolve(null); return }
      try { resolve(JSON.parse(await f.text())) }
      catch { reject(new Error(jsonErrorMsg)) }
    }
    input.click()
  })
}

type TaskKind = 'train' | 'tag' | 'reg' | 'download' | 'curate' | 'unknown'

const STATUS_TONE: Record<TaskStatus, string> = {
  pending:   'neutral',
  running:   'accent',
  done:      'ok',
  failed:    'err',
  canceled:  'neutral',
}

function inferKind(task: Task): TaskKind {
  const n = task.config_name.toLowerCase()
  if (n.includes('train') || n.includes('lora')) return 'train'
  if (n.includes('tag') || n.includes('caption') || n.includes('wd14')) return 'tag'
  if (n.includes('reg') || n.includes('regular')) return 'reg'
  if (n.includes('download') || n.includes('booru')) return 'download'
  if (n.includes('curate') || n.includes('filter')) return 'curate'
  return 'unknown'
}

function fmtAgo(ts: number): string {
  const sec = Math.max(0, Date.now() / 1000 - ts)
  if (sec < 60) return '刚刚'
  if (sec < 3600) return `${Math.floor(sec / 60)}m 前`
  if (sec < 86400) return `${Math.floor(sec / 3600)}h 前`
  return `${Math.floor(sec / 86400)}d 前`
}

function fmtDuration(start: number | null, end: number | null): string {
  if (!start) return '—'
  const e = end ?? Date.now() / 1000
  const sec = Math.max(0, e - start)
  if (sec < 60) return `${sec.toFixed(0)}s`
  const m = Math.floor(sec / 60); const s = Math.floor(sec % 60)
  if (m < 60) return `${m}m ${s}s`
  return `${Math.floor(m / 60)}h ${m % 60}m`
}

function fmtDurationShort(ms: number): string {
  if (ms < 60e3) return `${Math.round(ms / 1e3)}s`
  if (ms < 3600e3) return `${Math.round(ms / 60e3)}m`
  return `${(ms / 3600e3).toFixed(1)}h`
}

export default function QueuePage() {
  const { t } = useTranslation()
  const [tasks, setTasks] = useState<Task[]>([])
  const [loaded, setLoaded] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const reloadTimer = useRef<number | null>(null)
  const { toast } = useToast()
  const { confirm } = useDialog()
  const navigate = useNavigate()

  const STATUS_LABEL: Record<TaskStatus, string> = {
    pending:   t('status.queued'),
    running:   t('status.running'),
    done:      t('status.done'),
    failed:    t('status.failed'),
    canceled:  t('status.canceled'),
  }

  const KIND_LABEL: Record<TaskKind, string> = {
    train: t('nav.train'), tag: t('nav.tag'), reg: t('nav.reg'),
    download: t('nav.download'), curate: t('nav.curate'), unknown: t('monitor.taskLabel'),
  }
  const reload = useCallback(async () => {
    try { setTasks(await api.listQueue()); setError(null) }
    catch (e) { setError(String(e)) }
    finally { setLoaded(true) }
  }, [])

  useEventStream(
    (evt) => {
      if (evt.type === 'task_state_changed') {
        if (reloadTimer.current) return
        reloadTimer.current = window.setTimeout(() => {
          reloadTimer.current = null; void reload()
        }, 100)
      }
    },
    { onOpen: () => void reload() },
  )

  useEffect(() => { void reload() }, [reload])
  // 2s 时钟 tick：仅触发 re-render 让「23m ago」「elapsed 40m」之类的相对时间
  // 字段更新；不发任何 API。spread tasks 触发组件 re-render，下游 derived 状态
  // 跟着更新。
  useEffect(() => {
    const hasRunning = tasks.some((t) => t.status === 'running')
    if (!hasRunning) return
    const tick = window.setInterval(() => setTasks((ts) => [...ts]), 2000)
    return () => window.clearInterval(tick)
  }, [tasks])

  // 当前 running 任务的 id，给 monitor 进度条 / 状态卡片用。
  const runningTask = useMemo(
    () => tasks.find((t) => t.status === 'running') ?? null,
    [tasks],
  )
  const runningTaskId = runningTask?.id ?? null
  // monitor 进度走 useMonitorProgress hook (PR #37 增量协议)：runningTaskId
  // 切换时 hook 自动清状态 + 重拉 /api/state 冷启动；不需要本组件再写清理逻辑。
  const { state: monitor } = useMonitorProgress(runningTaskId)

  const clearDone = async () => {
    const done = tasks.filter((t) => t.status === 'done')
    if (done.length === 0) { toast(t('queue.noDone'), 'success'); return }
    if (!(await confirm(t('queue.confirmClearDone', { n: done.length }), { tone: 'danger', okText: t('common.delete') }))) return
    setBusy(true)
    try {
      for (const task of done) await api.deleteTask(task.id)
      toast(t('queue.cleared', { n: done.length }), 'success')
      await reload()
    } catch (e) { toast(String(e), 'error') }
    finally { setBusy(false) }
  }

  const sorted = useMemo(() => [...tasks].sort((a, b) => b.id - a.id), [tasks])

  const prevCount = useCallback((taskId: number): number => {
    let count = 0
    for (const t of sorted) {
      if (t.id === taskId) break
      if (t.status === 'running' || t.status === 'pending') count++
    }
    return count
  }, [sorted])

  const estimateEta = useCallback((task: Task): string | null => {
    if (task.status !== 'running' || !task.started_at) return null
    const elapsed = (Date.now() / 1000 - task.started_at) * 1000
    return `已运行 ${fmtDurationShort(elapsed)}`
  }, [])

  // 用 runningTask 派生比 tasks.some 再扫一遍便宜（runningTask 已经 memo 过）
  const hasRunning = runningTask !== null

  // 后端只有 per-task cancel，没有 queue-level pause。"暂停" 语义会让用户误以为
  // 可以恢复，但 cancelTask 是 terminal cancel（task 进 canceled 状态，重启从 0
  // 开始）；用 "取消" 语义对齐 QueueDetail 的 cancel 按钮。
  // 复用 CLI Ctrl+C 那套 save state / --resume-state 链路的 "真暂停" 是独立 feature
  // （见 memory/queue_pause_resume_via_sigint.md）。
  const cancelRunning = async () => {
    if (!runningTask) return
    const ok = await confirm(
      `取消当前任务 #${runningTask.id}？任务会在安全点停止，且无法恢复（重启训练会从 0 开始）。`,
      { tone: 'warn', okText: t('queue.cancelCurrent') },
    )
    if (!ok) return
    setBusy(true)
    try {
      await api.cancelTask(runningTask.id)
      toast(t('queueDetail.cancelSent'), 'success')
      await reload()
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setBusy(false)
    }
  }

  return (
    <StepShell
      idx={-1}
      title={t('queue.title')}
      subtitle={t('queue.description')}
      actions={
        <>
          <button onClick={clearDone} disabled={busy} className="btn btn-ghost btn-sm">{t('queue.clearDone')}</button>
          {hasRunning && (
            <button
              onClick={() => void cancelRunning()}
              disabled={busy}
              className="btn btn-secondary btn-sm text-warn border-warn"
              title={t('queue.cancelHint')}
            >
              {t('queue.cancelCurrent')}
            </button>
          )}
          <button
            disabled={busy || tasks.length === 0}
            onClick={async () => {
              try { await downloadJson(`queue_${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')}.json`, await api.exportQueue()) }
              catch (e) { setError(String(e)) }
            }}
            className="btn btn-ghost btn-sm"
          >{t('common.export')}</button>
          <button
            disabled={busy}
            onClick={async () => {
              let payload: unknown
              try { payload = await pickJsonFile(t('queue.jsonError')) }
              catch (e) { toast(String(e), 'error'); return }
              if (!payload) return
              setBusy(true)
              try {
                const r = await api.importQueue(payload)
                const renamedCount = Object.keys(r.renamed).length
                toast(t('queue.imported', { n: r.imported_count, renamed: renamedCount ? `（${renamedCount} 个改名）` : '' }), 'success')
                await reload()
              } catch (e) { setError(String(e)) }
              finally { setBusy(false) }
            }}
            className="btn btn-ghost btn-sm"
          >{t('common.import')}</button>
          <button onClick={() => void reload()} className="btn btn-ghost btn-sm">{t('common.refresh')}</button>
        </>
      }
    >
      <div className="flex flex-col gap-2.5">
        {error && (
          <div className="px-3.5 py-2.5 rounded-md bg-err-soft border border-err text-err text-xs font-mono">
            {error}
          </div>
        )}

        {!loaded ? (
          <div className="rounded-lg border border-subtle bg-surface overflow-hidden">
            {Array.from({ length: 3 }).map((_, i) => (
              <div
                key={i}
                className={`py-[18px] px-[22px] grid gap-4 items-center opacity-40 ${i < 2 ? 'border-b border-subtle' : 'border-b-0'}`}
                style={{ gridTemplateColumns: '60px 1fr 110px 1fr 160px' }}
              >
                <div className="h-3.5 rounded bg-overlay" />
                <div className="flex flex-col gap-1">
                  <div className="h-[13px] rounded bg-overlay w-3/5" />
                  <div className="h-2.5 rounded bg-overlay w-2/5" />
                </div>
                <div className="h-5 rounded bg-overlay" />
                <div className="h-2.5 rounded bg-overlay" />
                <div className="h-2.5 rounded bg-overlay" />
              </div>
            ))}
          </div>
        ) : tasks.length === 0 ? (
          <div className="rounded-lg border border-subtle bg-surface py-12 text-center">
            <div className="text-md font-semibold text-fg-secondary mb-1.5">
              {t('queue.empty')}
            </div>
            <div className="text-sm text-fg-tertiary">
              {t('queue.emptyHint')}
            </div>
          </div>
        ) : (
          <div className="flex flex-col gap-2">
            {sorted.map((task) => {
              const isRunning = task.status === 'running'
              const isTerminal = ['done', 'failed', 'canceled'].includes(task.status)
              const hasProject = !!(task.project_id && task.version_id)
              const kind = inferKind(task)
              const eta = estimateEta(task)
              const tone = STATUS_TONE[task.status]

              return (
                <button
                  key={task.id}
                  onClick={() => navigate(`/queue/${task.id}`)}
                  className={`card card-hover block overflow-hidden text-left p-0 ${isRunning ? 'cursor-pointer border border-accent bg-accent-soft' : 'cursor-default border border-subtle bg-surface'}`}
                >
                  <div
                    className="px-[22px] py-4 grid gap-4 items-center"
                    style={{ gridTemplateColumns: '60px 1fr 110px 1fr 160px' }}
                  >
                    <span className={`font-mono text-sm ${isRunning ? 'text-accent font-semibold' : 'text-fg-tertiary font-normal'}`}>
                      #{task.id}
                    </span>

                    <div style={{ minWidth: 0 }}>
                      <div className="font-semibold text-fg-primary text-sm overflow-hidden text-ellipsis whitespace-nowrap">
                        {task.name}
                      </div>
                      <div className="font-mono text-xs text-fg-tertiary mt-0.5 flex items-center gap-1.5">
                        <span>{KIND_LABEL[kind]}</span>
                        <span>{task.config_name}</span>
                        {hasProject && (
                          <Link
                            to={`/projects/${task.project_id}/v/${task.version_id}/train`}
                            onClick={(e) => e.stopPropagation()}
                            className="text-accent text-xs no-underline hover:underline shrink-0"
                          >
                            {t('queue.project')}
                          </Link>
                        )}
                      </div>
                    </div>

                    <span className={`badge badge-${tone} text-xs text-center`}>
                      {isRunning && <span className="dot dot-running" />}
                      {STATUS_LABEL[task.status]}
                    </span>

                    <div className="text-sm text-fg-secondary" style={{ minWidth: 0 }}>
                      {isRunning ? (
                        <div className="flex flex-col gap-0.5">
                          <span className="font-mono text-fg-tertiary text-xs">
                            {(() => {
                              if (
                                task.id === runningTaskId &&
                                monitor?.step != null &&
                                monitor.total_steps != null &&
                                monitor.total_steps > 0
                              ) {
                                return `step ${monitor.step.toLocaleString()} / ${monitor.total_steps.toLocaleString()}`
                              }
                              return fmtDuration(task.started_at, null)
                            })()}
                          </span>
                          <div className="h-1 bg-overlay rounded-sm overflow-hidden">
                            {(() => {
                              const haveSteps =
                                task.id === runningTaskId &&
                                monitor?.step != null &&
                                monitor.total_steps != null &&
                                monitor.total_steps > 0
                              if (haveSteps) {
                                const pct = Math.max(
                                  0,
                                  Math.min(100, (monitor!.step! / monitor!.total_steps!) * 100),
                                )
                                return <div className="h-full bg-accent rounded-sm" style={{ width: `${pct}%` }} />
                              }
                              return <div className="h-full bg-accent/40 rounded-sm animate-pulse" style={{ width: '20%' }} />
                            })()}
                          </div>
                        </div>
                      ) : task.error_msg ? (
                        <span className="text-err overflow-hidden text-ellipsis whitespace-nowrap block text-xs">
                          {task.error_msg}
                        </span>
                      ) : isTerminal ? (
                        <span className="font-mono text-fg-tertiary text-xs">
                          {t('queue.duration', { time: fmtDuration(task.started_at, task.finished_at) })}
                        </span>
                      ) : (
                        <span className="text-fg-tertiary text-xs">—</span>
                      )}
                    </div>

                    <span className="font-mono text-sm text-fg-tertiary text-right">
                      {isRunning ? (
                        <>
                          {eta && <span className="text-accent">{eta}</span>}
                          {eta && <br />}
                          <span className="text-xs">{fmtAgo(task.started_at!)} 开始</span>
                        </>
                      ) : task.finished_at ? (
                        <>
                          <span>{fmtAgo(task.finished_at)}</span>
                          <br />
                          <span className="text-xs text-fg-tertiary">{t('status.done')}</span>
                        </>
                      ) : (
                        <span>{t('queue.ahead', { n: prevCount(task.id) })}</span>
                      )}
                    </span>
                  </div>
                </button>
              )
            })}
          </div>
        )}
      </div>
    </StepShell>
  )
}
