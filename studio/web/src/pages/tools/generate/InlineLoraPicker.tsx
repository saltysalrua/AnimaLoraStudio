import { useEffect, useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import type { ProjectLora } from './types'
import { ckptStemFromPath } from './xy'

/** 项目缩写图标（2 字符 uppercase，从 title 提字母数字派生）。 */
export function projectAbbr(title: string): string {
  const cleaned = title.replace(/[^a-zA-Z0-9]/g, '')
  return (cleaned.slice(0, 2) || '??').toUpperCase()
}

function ProjectIcon({ title }: { title: string }) {
  return (
    <div className="shrink-0 w-7 h-7 rounded bg-sunken text-fg-tertiary text-2xs font-mono flex items-center justify-center border border-subtle">
      {projectAbbr(title)}
    </div>
  )
}

/** 内嵌 LoRA 多选挑选器：扁平列表 + 搜索 + 一键清空 + 短名显示。
 *
 * 列表每行：[项目 icon] [项目 / 版本] [训练中 pill] [短名] [✓ / +]
 * 不再按 project 分组（用户反馈分组冗余）；显示 stem 名（去 .safetensors）。
 *
 * onRemove 可选：传了就支持点击已添加项取消勾选（多选 toggle）。
 * onClearAll 可选：传了就显示一键清空按钮。
 */
export default function InlineLoraPicker({
  projectLoras, selectedPaths,
  onPick, onRemove, onClearAll,
  onClose, onPickExternal,
}: {
  projectLoras: ProjectLora[]
  selectedPaths: Set<string>
  onPick: (path: string) => void
  onRemove?: (path: string) => void
  onClearAll?: () => void
  onClose: () => void
  onPickExternal: () => void
}) {
  const { t } = useTranslation()
  const [search, setSearch] = useState('')
  const [projectId, setProjectId] = useState<string>('all')
  const [versionKey, setVersionKey] = useState<string>('all')

  const projects = useMemo(() => {
    const seen = new Map<number, string>()
    for (const l of projectLoras) {
      if (!seen.has(l.projectId)) seen.set(l.projectId, l.projectTitle)
    }
    return Array.from(seen.entries()).map(([id, title]) => ({ id, title }))
  }, [projectLoras])

  const versions = useMemo(() => {
    const seen = new Map<string, ProjectLora>()
    for (const l of projectLoras) {
      if (projectId !== 'all' && l.projectId !== Number(projectId)) continue
      const key = `${l.projectId}:${l.versionId}`
      if (!seen.has(key)) seen.set(key, l)
    }
    return Array.from(seen.entries()).map(([key, l]) => ({ key, item: l }))
  }, [projectId, projectLoras])

  // versions 变化（切项目 / 上游 projectLoras 增减）时，若当前 versionKey 不再
  // 出现在新 versions 中，回退到 'all'。切项目时 onChange 已显式 reset，这条 effect
  // 是 projectLoras 列表外部变化（新增 / 删除 LoRA）的兜底，保留两路是有意为之。
  useEffect(() => {
    if (versionKey === 'all') return
    if (!versions.some((v) => v.key === versionKey)) setVersionKey('all')
  }, [versionKey, versions])

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase()
    return projectLoras.filter((l) => {
      if (projectId !== 'all' && l.projectId !== Number(projectId)) return false
      if (versionKey !== 'all' && `${l.projectId}:${l.versionId}` !== versionKey) return false
      if (!q) return true
      return (
        l.projectTitle.toLowerCase().includes(q) ||
        l.versionLabel.toLowerCase().includes(q) ||
        ckptStemFromPath(l.path).toLowerCase().includes(q)
      )
    })
  }, [projectId, projectLoras, search, versionKey])

  const selectedCount = selectedPaths.size

  return (
    <div
      className="rounded-md border border-subtle bg-overlay p-2.5 flex flex-col gap-2"
      data-testid="inline-lora-picker"
    >
      {/* 两行布局（P1-H）：第 1 行 = 搜索 + 操作；第 2 行 = 项目 / 版本筛选。
          原来挤一行 7 个元素在 picker 宽 < 600px 时 wrap 成 2-3 行视觉零散。
          项目数 ≤ 1 时直接隐藏筛选行（dropdown 无意义）。 */}
      <div className="flex items-center gap-2">
        <input
          type="text"
          className="input flex-1 text-xs min-w-[160px]"
          placeholder={t('generate.searchProjectVersionFile')}
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          autoFocus
        />
        <span className="text-2xs text-fg-tertiary whitespace-nowrap">
          {t('generate.selectedCount', { selected: selectedCount, total: filtered.length })}
        </span>
        {onClearAll && selectedCount > 0 && (
          <button
            onClick={onClearAll}
            className="btn btn-ghost btn-sm text-2xs text-fg-tertiary"
            title={t('generate.clearAllLoraTitle')}
          >
            {t('generate.clearAll')}
          </button>
        )}
        <button
          onClick={onPickExternal}
          className="btn btn-ghost btn-sm text-2xs text-fg-tertiary"
          title={t('generate.pickExternalTitle')}
        >
          {t('generate.externalFile')}
        </button>
        <button
          onClick={onClose}
          className="btn btn-ghost btn-sm text-fg-tertiary px-1.5"
          title={t('common.close')}
          aria-label={t('generate.closePicker')}
        >
          ×
        </button>
      </div>
      {projects.length > 1 && (
        <div className="flex items-center gap-2">
          <select
            className="input text-xs"
            value={projectId}
            onChange={(e) => {
              setProjectId(e.target.value)
              setVersionKey('all')
            }}
            aria-label={t('generate.filterProject')}
            style={{ width: 132 }}
          >
            <option value="all">{t('generate.allProjects')}</option>
            {projects.map((p) => (
              <option key={p.id} value={p.id}>{p.title}</option>
            ))}
          </select>
          <select
            className="input text-xs"
            value={versionKey}
            onChange={(e) => setVersionKey(e.target.value)}
            aria-label={t('generate.filterVersion')}
            disabled={versions.length === 0}
            style={{ width: 150 }}
          >
            <option value="all">{t('generate.allVersions')}</option>
            {versions.map(({ key, item }) => (
              <option key={key} value={key}>
                {projectId === 'all'
                  ? `${item.projectTitle} / ${item.versionLabel}`
                  : item.versionLabel}
              </option>
            ))}
          </select>
        </div>
      )}

      {/* 列表：扁平，每行一个 LoRA */}
      <div
        className="flex flex-col gap-px overflow-y-auto"
        style={{ maxHeight: 360 }}
        data-testid="inline-lora-list"
      >
        {filtered.map((l) => {
          const added = selectedPaths.has(l.path)
          const stem = ckptStemFromPath(l.path)
          const handleClick = () => {
            if (added) {
              if (onRemove) onRemove(l.path)
            } else {
              onPick(l.path)
            }
          }
          return (
            <button
              key={`${l.projectId}-${l.versionId}`}
              onClick={handleClick}
              disabled={added && !onRemove}
              className={`flex items-center gap-2 px-2 py-1.5 rounded text-xs text-left border-none transition-colors ${
                added
                  ? (onRemove
                      ? 'bg-accent-soft text-accent cursor-pointer hover:opacity-80'
                      : 'bg-sunken text-fg-tertiary cursor-not-allowed')
                  : 'bg-transparent hover:bg-surface text-fg-secondary cursor-pointer'
              }`}
              style={{
                background: added
                  ? 'var(--accent-soft)'
                  : undefined,
              }}
            >
              <ProjectIcon title={l.projectTitle} />
              <div className="flex-1 min-w-0 flex flex-col gap-px">
                <div className="font-medium truncate flex items-center gap-1.5">
                  <span>{l.projectTitle} / {l.versionLabel}</span>
                  {l.stage === 'training' && (
                    <span className="badge badge-info" style={{ fontSize: 10 }}>{t('status.training')}</span>
                  )}
                </div>
                <div className="text-2xs text-fg-tertiary font-mono truncate" title={l.path}>
                  {stem}
                </div>
              </div>
              <span className="font-mono text-2xs shrink-0" style={{ minWidth: 16, textAlign: 'right' }}>
                {added ? '✓' : '+'}
              </span>
            </button>
          )
        })}

        {filtered.length === 0 && (
          <div className="text-fg-tertiary text-xs px-2 py-4 text-center">
            {search ? t('generate.noMatchingLora') : t('generate.noTrainedLora')}
          </div>
        )}
      </div>
    </div>
  )
}
