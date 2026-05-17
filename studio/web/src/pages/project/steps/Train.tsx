import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useNavigate, useOutletContext } from 'react-router-dom'
import {
  api,
  type ConfigData,
  type PresetSummary,
  type ProjectDetail,
  type RegStatus,
  type SchemaResponse,
  type Version,
  type VersionConfigResponse,
} from '../../../api/client'
import ConfigSkeleton from '../../../components/ConfigSkeleton'
import { useDialog } from '../../../components/Dialog'
import SchemaForm from '../../../components/SchemaForm'
import StepShell from '../../../components/StepShell'
import { useToast } from '../../../components/Toast'
import { useAdvancedMode } from '../../../lib/useAdvancedMode'
import {
  PRESET_NAME_RE,
  defaultsFromSchema,
  loadPresetDescriptions,
  savePresetDescriptions,
} from '../../../lib/preset-helpers'

const GLOBAL_MODEL_FIELDS = [
  'transformer_path',
  'vae_path',
  'text_encoder_path',
  't5_tokenizer_path',
]

interface Ctx {
  project: ProjectDetail
  activeVersion: Version | null
  reload: () => Promise<void>
}

export default function TrainPage() {
  const { t } = useTranslation()
  const { project, activeVersion, reload } = useOutletContext<Ctx>()
  const { toast } = useToast()
  const { confirm, prompt } = useDialog()
  const navigate = useNavigate()

  const [schema, setSchema] = useState<SchemaResponse | null>(null)
  const [presets, setPresets] = useState<PresetSummary[]>([])
  const [configResp, setConfigResp] = useState<VersionConfigResponse | null>(null)
  const [config, setConfig] = useState<ConfigData | null>(null)
  const [reg, setReg] = useState<RegStatus | null>(null)
  const [busy, setBusy] = useState(false)

  const savedJsonRef = useRef<string | null>(null)
  const configRef = useRef<ConfigData | null>(null)
  const inFlightSaveRef = useRef<Promise<void> | null>(null)
  const debounceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const presetBaselineRef = useRef<ConfigData | null>(null)

  const [pickerOpen, setPickerOpen] = useState(false)
  const [pickerSearch, setPickerSearch] = useState('')
  const [advancedMode, toggleAdvancedMode] = useAdvancedMode()
  const pickerAnchorRef = useRef<HTMLButtonElement | null>(null)
  const pickerPopRef = useRef<HTMLDivElement | null>(null)

  const [creatingPreset, setCreatingPreset] = useState(false)
  const [newPresetName, setNewPresetName] = useState('')
  const [newPresetDesc, setNewPresetDesc] = useState('')
  const [newPresetConfig, setNewPresetConfig] = useState<ConfigData | null>(null)
  const [newNameError, setNewNameError] = useState('')

  const setConfigSync = useCallback((v: ConfigData | null) => {
    configRef.current = v
    setConfig(v)
  }, [])

  const vid = activeVersion?.id ?? null

  const refreshConfig = useCallback(async () => {
    if (!vid) return
    try {
      const r = await api.getVersionConfig(project.id, vid)
      setConfigResp(r)
      setConfigSync(r.config)
      savedJsonRef.current = JSON.stringify(r.config)
    } catch (e) {
      toast(t('train.loadConfigFailed', { error: e }), 'error')
    }
  }, [project.id, vid, toast, setConfigSync, t])

  const refreshPresetBaseline = useCallback(async (name: string | null) => {
    if (!name) { presetBaselineRef.current = null; return }
    try {
      presetBaselineRef.current = await api.getPreset(name)
    } catch {
      presetBaselineRef.current = null
    }
  }, [])

  useEffect(() => {
    void refreshPresetBaseline(activeVersion?.config_name ?? null)
  }, [activeVersion?.config_name, refreshPresetBaseline])

  useEffect(() => {
    api.schema().then(setSchema).catch((e) => toast(t('train.loadSchemaFailed', { error: e }), 'error'))
    api.listPresets().then(setPresets).catch(() => setPresets([]))
  }, [toast, t])

  useEffect(() => {
    void refreshConfig()
  }, [refreshConfig])

  useEffect(() => {
    if (!vid) return
    api.getRegStatus(project.id, vid).then(setReg).catch(() => setReg(null))
  }, [project.id, vid])

  const disabledFields = GLOBAL_MODEL_FIELDS
  const disabledHints = useMemo(() => {
    const h: Record<string, string> = {}
    for (const f of GLOBAL_MODEL_FIELDS) h[f] = t('train.globalAutoHint')
    return h
  }, [t])
  const autoHints = useMemo(() => {
    const h: Record<string, string> = {}
    for (const f of configResp?.project_specific_fields ?? []) {
      if (!GLOBAL_MODEL_FIELDS.includes(f)) h[f] = t('train.projectAutoHint')
    }
    return h
  }, [configResp?.project_specific_fields, t])

  const stripProjectFields = useCallback((cfg: ConfigData | null): string => {
    if (!cfg) return ''
    const skip = new Set(configResp?.project_specific_fields ?? [])
    const filtered: ConfigData = {}
    for (const k of Object.keys(cfg)) {
      if (!skip.has(k)) filtered[k] = cfg[k]
    }
    return JSON.stringify(filtered)
  }, [configResp?.project_specific_fields])

  const customized = useMemo(() => {
    if (!config || !presetBaselineRef.current) return false
    return stripProjectFields(config) !== stripProjectFields(presetBaselineRef.current)
  }, [config, stripProjectFields])

  const persistConfig = useCallback(async (cfg: ConfigData): Promise<void> => {
    while (inFlightSaveRef.current) {
      await inFlightSaveRef.current
    }
    if (JSON.stringify(cfg) === savedJsonRef.current) return
    const p = (async () => {
      const r = await api.putVersionConfig(project.id, vid!, cfg)
      setConfigResp((prev) => prev ? { ...prev, has_config: true, config: r.config } : prev)
      savedJsonRef.current = JSON.stringify(r.config)
      if (configRef.current === cfg) {
        configRef.current = r.config
        setConfig(r.config)
      }
    })()
    inFlightSaveRef.current = p
    try { await p } finally { inFlightSaveRef.current = null }
  }, [project.id, vid])

  useEffect(() => {
    if (!config) return
    if (JSON.stringify(config) === savedJsonRef.current) return
    debounceTimerRef.current = setTimeout(() => {
      debounceTimerRef.current = null
      void persistConfig(config).catch((e) => toast(t('train.saveFailed', { error: e }), 'error'))
    }, 600)
    return () => {
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current)
        debounceTimerRef.current = null
      }
    }
  }, [config, persistConfig, toast, t])

  useEffect(() => {
    return () => {
      const cur = configRef.current
      if (!cur || !vid) return
      if (JSON.stringify(cur) === savedJsonRef.current) return
      void api.putVersionConfig(project.id, vid, cur).catch(() => {})
    }
  }, [project.id, vid])

  const filteredPresets = useMemo(
    () => presets.filter((p) => !pickerSearch || p.name.toLowerCase().includes(pickerSearch.toLowerCase())),
    [presets, pickerSearch],
  )

  useEffect(() => {
    if (!pickerOpen) return
    const onDocClick = (e: MouseEvent) => {
      const target = e.target as Node
      if (pickerPopRef.current?.contains(target) || pickerAnchorRef.current?.contains(target)) return
      setPickerOpen(false)
    }
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setPickerOpen(false) }
    document.addEventListener('mousedown', onDocClick)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDocClick)
      document.removeEventListener('keydown', onKey)
    }
  }, [pickerOpen])

  if (!activeVersion || !vid) {
    return <p className="text-fg-tertiary p-6">{t('train.noVersion')}</p>
  }

  const onForkPreset = async (name: string) => {
    if (!name) return
    if (configResp?.has_config) {
      const ok = await confirm(t('train.confirmForkPreset'), { tone: 'warn', okText: t('train.forkPresetOkText') })
      if (!ok) return
    }
    setBusy(true)
    try {
      await api.forkPresetForVersion(project.id, vid, name)
      await refreshConfig()
      toast(t('train.forkedFrom', { name }), 'success')
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setBusy(false)
    }
  }

  const onSaveAsPreset = async () => {
    const name = await prompt(t('train.promptPresetName'), {
      placeholder: 'my-preset',
      validate: (v) => {
        const trimmed = v.trim()
        if (!trimmed) return t('train.nameEmpty')
        if (!PRESET_NAME_RE.test(trimmed)) return t('train.nameInvalid')
        return null
      },
    })
    if (!name) return
    const trimmed = name.trim()
    setBusy(true)
    try {
      await api.saveVersionConfigAsPreset(project.id, vid, trimmed, false)
      const list = await api.listPresets()
      setPresets(list)
      toast(t('train.savedAsPreset', { name: trimmed }), 'success')
    } catch (e) {
      const msg = String(e)
      if (msg.includes('已存在')) {
        const overwrite = await confirm(t('train.alreadyExists', { name: trimmed }), { tone: 'danger', okText: t('train.overwriteOkText') })
        if (overwrite) {
          try {
            await api.saveVersionConfigAsPreset(project.id, vid, trimmed, true)
            const list = await api.listPresets()
            setPresets(list)
            toast(t('train.overwritePreset', { name: trimmed }), 'success')
          } catch (e2) {
            toast(String(e2), 'error')
          }
        }
      } else {
        toast(msg, 'error')
      }
    } finally {
      setBusy(false)
    }
  }

  const defaultPresetName = (): string => {
    if (!activeVersion) return project.slug
    const candidate = `${project.slug}_${activeVersion.label}`
    if (PRESET_NAME_RE.test(candidate)) return candidate
    return `${project.slug}_v${activeVersion.id}`
  }

  const startCreatePreset = () => {
    setPickerOpen(false)
    setNewPresetName(defaultPresetName())
    setNewPresetDesc('')
    setNewPresetConfig(defaultsFromSchema(schema))
    setNewNameError('')
    setCreatingPreset(true)
  }

  const cancelCreatePreset = () => {
    setCreatingPreset(false)
    setNewNameError('')
  }

  const saveNewPreset = async () => {
    const name = newPresetName.trim()
    if (!name) { setNewNameError(t('train.nameEmpty')); return }
    if (!PRESET_NAME_RE.test(name)) { setNewNameError(t('train.nameInvalid')); return }
    if (!newPresetConfig || !vid) return
    if (presets.some((p) => p.name === name)) {
      const overwrite = await confirm(t('train.alreadyExists', { name }), { tone: 'danger', okText: t('train.overwriteOkText') })
      if (!overwrite) return
    }
    setBusy(true)
    try {
      await api.savePreset(name, newPresetConfig)
      const desc = newPresetDesc.trim()
      if (desc) {
        const all = loadPresetDescriptions()
        all[name] = desc
        savePresetDescriptions(all)
      }
      const list = await api.listPresets()
      setPresets(list)
      await api.forkPresetForVersion(project.id, vid, name)
      await refreshConfig()
      void refreshPresetBaseline(name)
      setCreatingPreset(false)
      toast(t('train.createdPreset', { name }), 'success')
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setBusy(false)
    }
  }

  const onEnqueue = async () => {
    if (!configResp?.has_config) { toast(t('train.noPresetError'), 'error'); return }
    setBusy(true)
    try {
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current)
        debounceTimerRef.current = null
      }
      if (inFlightSaveRef.current) await inFlightSaveRef.current
      const cur = configRef.current
      if (cur && JSON.stringify(cur) !== savedJsonRef.current) {
        await persistConfig(cur)
      }
      const task = await api.enqueueVersionTraining(project.id, vid)
      toast(t('train.enqueuedNav', { id: task.id }), 'success')
      void reload()
      navigate('/queue')
    } catch (e) {
      toast(String(e), 'error')
    } finally {
      setBusy(false)
    }
  }

  return (
    <StepShell
      idx={6}
      title={t('steps.train.title')}
      subtitle={t('steps.train.subtitle')}
      actions={
        <button
          onClick={() => void onEnqueue()}
          disabled={busy || !configResp?.has_config}
          className="btn btn-primary"
        >
          {t('train.startTrainBtn')}
        </button>
      }
    >
      <div className="flex flex-col h-full gap-3">

        <div className="grid grid-cols-[1.5fr_1fr] gap-3 flex-1 min-h-0">

          {/* 左栏 */}
          <div className="flex flex-col gap-3 min-h-0 min-w-0 overflow-y-auto">

          {/* 预设 picker */}
          <section className="flex items-center gap-2.5 shrink-0 relative">
            <button
              ref={pickerAnchorRef}
              onClick={() => { setPickerOpen((v) => !v); setPickerSearch('') }}
              disabled={busy}
              className={[
                'flex items-center gap-3 min-w-[300px] pl-3.5 pr-3 py-2.5',
                'rounded-md border transition-[border-color,background] duration-100',
                pickerOpen ? 'border-accent bg-accent-soft' : 'border-dim bg-surface shadow-sm hover:border-bold',
                busy ? 'cursor-default' : 'cursor-pointer',
              ].join(' ')}
              title={t('train.presetLabel')}
            >
              <span className="text-[10px] uppercase tracking-[0.08em] text-fg-tertiary font-semibold">
                {t('train.presetLabel')}
              </span>
              <span className={[
                'font-mono text-md font-semibold flex-1 text-left truncate',
                configResp?.has_config ? 'text-fg-primary' : 'text-fg-tertiary',
              ].join(' ')}>
                {activeVersion.config_name ?? t('train.noPreset')}
                {customized && (
                  <span className="ml-2 text-xs text-warn font-normal" title={t('train.customizedTitle')}>
                    {t('train.customized')}
                  </span>
                )}
              </span>
              <span className="text-fg-tertiary text-md">▾</span>
            </button>
            <button
              onClick={() => void onSaveAsPreset()}
              disabled={busy || !configResp?.has_config}
              className="btn btn-ghost btn-sm"
              title={t('train.saveAsPresetTitle')}
            >
              {t('train.saveAsPreset')}
            </button>

            {pickerOpen && (
              <div
                ref={pickerPopRef}
                role="dialog"
                aria-label={t('train.presetLabel')}
                className="absolute top-[calc(100%+6px)] left-0 w-[480px] max-h-[480px] overflow-hidden rounded-md border border-subtle bg-surface shadow-lg flex flex-col z-50"
              >
                <div className="p-2.5 border-b border-subtle flex items-center gap-2">
                  <span className="relative flex-1 inline-flex items-center">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                      strokeWidth="2" strokeLinecap="round"
                      className="absolute left-2 text-fg-tertiary pointer-events-none">
                      <circle cx="11" cy="11" r="7"/><path d="m21 21-4.3-4.3"/>
                    </svg>
                    <input
                      autoFocus
                      className="input w-full pl-7 text-sm"
                      placeholder={t('train.filterPresets')}
                      value={pickerSearch}
                      onChange={(e) => setPickerSearch(e.target.value)}
                    />
                  </span>
                </div>

                <div className="flex-1 min-h-0 overflow-y-auto p-2.5">
                  <div className="grid grid-cols-2 gap-2">
                    {!pickerSearch && (
                      <button
                        onClick={startCreatePreset}
                        disabled={busy}
                        className={[
                          'rounded-sm px-2.5 py-2 text-left border border-dashed transition-colors',
                          'border-subtle text-accent hover:border-accent hover:bg-accent-soft',
                          busy ? 'cursor-default' : 'cursor-pointer',
                          'bg-transparent text-sm font-semibold',
                        ].join(' ')}
                      >
                        {t('train.newPreset')}
                      </button>
                    )}
                    {filteredPresets.map((p) => {
                      const active = p.name === activeVersion.config_name
                      return (
                        <button
                          key={p.name}
                          onClick={() => { setPickerOpen(false); void onForkPreset(p.name) }}
                          disabled={busy}
                          className={[
                            'rounded-sm px-2.5 py-2 text-left border transition-colors',
                            active ? 'border-accent bg-accent-soft' : 'border-subtle bg-sunken hover:border-bold',
                            busy ? 'cursor-default' : 'cursor-pointer',
                          ].join(' ')}
                        >
                          <div className={['text-sm font-mono font-semibold truncate', active ? 'text-accent' : 'text-fg-primary'].join(' ')}>
                            {p.name}
                          </div>
                          <div className="text-xs text-fg-tertiary mt-0.5">
                            {active ? t('train.presetCurrent') : t('train.presetApply')}
                          </div>
                        </button>
                      )
                    })}
                  </div>
                  {presets.length > 0 && filteredPresets.length === 0 && (
                    <div className="text-fg-tertiary text-sm text-center py-4">
                      {t('train.noMatch', { search: pickerSearch })}
                    </div>
                  )}
                </div>
              </div>
            )}
          </section>

            {creatingPreset && schema && newPresetConfig ? (
              <section className="flex-1 min-h-0 overflow-y-auto pr-1">
                <div className="flex flex-col gap-3">
                  <div className="rounded-md border border-subtle bg-surface px-3.5 py-2.5">
                    <div className="flex gap-2.5">
                      <label className="flex-1 flex flex-col gap-1">
                        <span className="text-sm font-medium text-fg-secondary">{t('train.presetName')}</span>
                        <input
                          autoFocus
                          className="input input-mono font-mono"
                          placeholder="my-training-preset"
                          value={newPresetName}
                          onChange={(e) => { setNewPresetName(e.target.value); setNewNameError('') }}
                          disabled={busy}
                        />
                        {newNameError && <span className="text-xs text-err">{newNameError}</span>}
                      </label>
                      <label className="flex-[1.5] flex flex-col gap-1">
                        <span className="text-sm font-medium text-fg-secondary">{t('train.presetDesc')}</span>
                        <input
                          className="input"
                          placeholder={t('train.descPlaceholder')}
                          value={newPresetDesc}
                          onChange={(e) => setNewPresetDesc(e.target.value)}
                          disabled={busy}
                        />
                      </label>
                    </div>
                  </div>
                  <div className="rounded-md border border-subtle bg-surface px-3.5 py-2.5">
                    <SchemaForm
                      schema={schema}
                      values={newPresetConfig}
                      onChange={setNewPresetConfig}
                    />
                  </div>
                  <div className="flex gap-2 shrink-0">
                    <button onClick={() => void saveNewPreset()} disabled={busy} className="btn btn-primary">
                      {busy ? t('train.savingBtn') : t('train.createAndApply')}
                    </button>
                    <button onClick={cancelCreatePreset} disabled={busy} className="btn btn-ghost">
                      {t('common.cancel')}
                    </button>
                  </div>
                </div>
              </section>
            ) : configResp === null || !schema ? (
              <ConfigSkeleton label={t('train.loadingConfig')} />
            ) : !configResp.has_config ? (
              <div className="flex-1 flex items-center justify-center text-fg-tertiary text-sm rounded-md border border-dashed border-dim">
                {t('train.noConfigHint')}
              </div>
            ) : config ? (
              <section className="flex-1 min-h-0 overflow-y-auto pr-1">
                <div className="flex justify-end mb-2">
                  <div className="inline-flex rounded-md border border-subtle overflow-hidden text-xs">
                    <button
                      type="button"
                      onClick={() => !advancedMode || toggleAdvancedMode()}
                      className={`px-3 py-1 transition-colors ${!advancedMode ? 'bg-accent text-white' : 'bg-surface text-fg-secondary hover:bg-subtle'}`}
                    >
                      {t('train.simpleMode')}
                    </button>
                    <button
                      type="button"
                      onClick={() => advancedMode || toggleAdvancedMode()}
                      className={`px-3 py-1 transition-colors ${advancedMode ? 'bg-accent text-white' : 'bg-surface text-fg-secondary hover:bg-subtle'}`}
                    >
                      {t('train.advancedMode')}
                    </button>
                  </div>
                </div>
                <SchemaForm
                  schema={schema}
                  values={config}
                  onChange={setConfigSync}
                  disabledFields={disabledFields}
                  disabledHints={disabledHints}
                  autoHints={autoHints}
                  advancedMode={advancedMode}
                />
              </section>
            ) : (
              <ConfigSkeleton label={t('train.loadingConfig')} />
            )}
          </div>

        {/* 右栏：训练集 + 正则集分布 */}
        <DatasetStatsPanel activeVersion={activeVersion} reg={reg} config={config} />
      </div>
    </div>
    </StepShell>
  )
}

function parseFolderRepeat(name: string): { repeat: number; label: string } {
  const m = name.match(/^(\d+)_(.*)$/)
  if (m) return { repeat: parseInt(m[1], 10), label: m[2] }
  return { repeat: 1, label: name }
}

function aggregateRegFolders(files: string[]): Array<{ name: string; image_count: number }> {
  const m = new Map<string, number>()
  for (const f of files) {
    const idx = f.indexOf('/')
    if (idx < 0) continue
    const folder = f.slice(0, idx)
    m.set(folder, (m.get(folder) ?? 0) + 1)
  }
  return Array.from(m.entries()).map(([name, image_count]) => ({ name, image_count })).sort((a, b) => a.name.localeCompare(b.name))
}

function DatasetStatsPanel({
  activeVersion, reg, config,
}: {
  activeVersion: Version | null
  reg: RegStatus | null
  config: ConfigData | null
}) {
  const { t } = useTranslation()
  const trainFolders = activeVersion?.stats?.train_folders ?? []
  const regFolders = useMemo(
    () => (reg && reg.exists ? aggregateRegFolders(reg.files) : []),
    [reg]
  )

  const trainEffective = trainFolders.reduce(
    (s, f) => s + parseFolderRepeat(f.name).repeat * f.image_count, 0,
  )
  const regEffective = regFolders.reduce(
    (s, f) => s + parseFolderRepeat(f.name).repeat * f.image_count, 0,
  )
  const totalEffective = trainEffective + regEffective

  const bs = Number(config?.batch_size) || 1
  const ga = Number(config?.grad_accum) || 1
  const epochs = Number(config?.epochs) || 0
  const maxSteps = Number(config?.max_steps) || 0
  const stepsPerEpoch = totalEffective > 0 ? Math.ceil(totalEffective / (bs * ga)) : null
  const naturalTotal = stepsPerEpoch !== null && epochs > 0 ? stepsPerEpoch * epochs : null
  const finalTotal = naturalTotal !== null && maxSteps > 0 ? Math.min(maxSteps, naturalTotal) : naturalTotal
  const maxStepsTruncates = maxSteps > 0 && naturalTotal !== null && maxSteps < naturalTotal

  return (
    <div className="flex flex-col gap-3 min-w-0">
      <div className="rounded-md border border-subtle bg-surface px-3 py-2.5">
        <div className="flex items-center gap-1.5 mb-2.5">
          <span className="inline-block w-1.5 h-1.5 rounded-full bg-accent shrink-0" />
          <span className="caption uppercase tracking-[0.06em] text-xs">{t('train.statsTitle')}</span>
        </div>

        <FolderSection
          title="train/"
          folders={trainFolders}
          effective={trainEffective}
          empty={t('train.noTrainImages')}
        />

        <div className="h-2" />

        <FolderSection
          title="reg/"
          folders={regFolders}
          effective={regEffective}
          empty={reg && !reg.exists ? t('train.regNotBuilt') : t('train.noRegImages')}
        />

        <div className="mt-2.5 pt-2 border-t border-subtle flex flex-col gap-1 text-xs">
          <Row label={t('train.effectiveSamples')} value={String(totalEffective)} bold />
          {stepsPerEpoch !== null && (
            <Row label={`÷ batch × ga (${bs} × ${ga})`} value={`≈ ${stepsPerEpoch} steps/epoch`} dim />
          )}
          {naturalTotal !== null && (
            <Row label={`× epochs (${epochs})`} value={`≈ ${naturalTotal} steps`} dim />
          )}
          {finalTotal !== null && (
            <Row
              label={maxStepsTruncates ? t('train.maxStepsLabel', { n: maxSteps }) : t('train.totalSteps')}
              value={`≈ ${finalTotal}`}
              bold
            />
          )}
        </div>
      </div>
    </div>
  )
}

function FolderSection({
  title, folders, effective, empty,
}: {
  title: string
  folders: Array<{ name: string; image_count: number }>
  effective: number
  empty: string
}) {
  return (
    <div>
      <div className="flex items-baseline justify-between text-xs mb-1">
        <span className="font-mono text-fg-secondary font-medium">{title}</span>
        {folders.length > 0 && <span className="font-mono text-fg-tertiary">∑ {effective}</span>}
      </div>
      {folders.length === 0 ? (
        <div className="text-xs text-fg-tertiary pl-1">{empty}</div>
      ) : (
        <div className="flex flex-col gap-0.5">
          {folders.map((f) => {
            const { repeat, label } = parseFolderRepeat(f.name)
            const eff = repeat * f.image_count
            return (
              <div
                key={f.name}
                className="flex items-baseline gap-1.5 text-xs font-mono text-fg-secondary pl-1"
                title={`${f.name}：${repeat} repeat × ${f.image_count} = ${eff}`}
              >
                <span className="text-fg-tertiary">{label}</span>
                <span className="flex-1 border-b border-dotted border-subtle self-end mb-1" />
                <span>
                  <span className="text-accent">{repeat}</span>
                  <span className="text-fg-tertiary"> × </span>
                  <span className="text-fg-primary">{f.image_count}</span>
                  <span className="text-fg-tertiary"> = </span>
                  <span className="text-fg-primary font-semibold">{eff}</span>
                </span>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

function Row({ label, value, bold, dim }: { label: string; value: string; bold?: boolean; dim?: boolean }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
      <span style={{ color: dim ? 'var(--fg-tertiary)' : 'var(--fg-secondary)' }}>{label}</span>
      <span style={{
        fontFamily: 'var(--font-mono)',
        color: bold ? 'var(--accent)' : dim ? 'var(--fg-tertiary)' : 'var(--fg-primary)',
        fontWeight: bold ? 700 : 500,
      }}>{value}</span>
    </div>
  )
}
