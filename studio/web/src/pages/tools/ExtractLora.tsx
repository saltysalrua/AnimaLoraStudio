import { useMemo, useState, type InputHTMLAttributes } from 'react'
import { api, type ExtractLoraRequest, type ExtractLoraResult } from '../../api/client'
import PageHeader from '../../components/PageHeader'
import PathPicker from '../../components/PathPicker'
import { useToast } from '../../components/Toast'

const DEFAULT_PATTERN = '*q_proj.weight|*k_proj.weight|*v_proj.weight|*output_proj.weight|*mlp.layer1.weight|*mlp.layer2.weight'

type PathField = 'base_path' | 'tuned_path' | 'output_path'

interface FormState {
  base_path: string
  tuned_path: string
  output_path: string
  rank: string
  alpha: string
  target_pattern: string
  prefix: string
}

const initialForm: FormState = {
  base_path: '',
  tuned_path: '',
  output_path: 'studio_data/extracted_lora.safetensors',
  rank: '32',
  alpha: '',
  target_pattern: DEFAULT_PATTERN,
  prefix: 'lora_unet',
}

export default function ExtractLoraPage() {
  const { toast } = useToast()
  const [form, setForm] = useState<FormState>(initialForm)
  const [result, setResult] = useState<ExtractLoraResult | null>(null)
  const [busy, setBusy] = useState<'validate' | 'extract' | null>(null)
  const [picker, setPicker] = useState<PathField | null>(null)

  const requestBody = useMemo<ExtractLoraRequest>(() => ({
    base_path: form.base_path.trim(),
    tuned_path: form.tuned_path.trim(),
    output_path: form.output_path.trim(),
    rank: Number(form.rank),
    alpha: form.alpha.trim() ? Number(form.alpha) : null,
    target_pattern: form.target_pattern.trim(),
    prefix: form.prefix.trim() || 'lora_unet',
  }), [form])

  const canSubmit = Boolean(
    requestBody.base_path &&
    requestBody.tuned_path &&
    requestBody.output_path &&
    Number.isFinite(requestBody.rank) &&
    requestBody.rank >= 1 &&
    (requestBody.alpha == null || (Number.isFinite(requestBody.alpha) && requestBody.alpha > 0)) &&
    !busy,
  )

  const setField = (key: keyof FormState, value: string) => {
    setForm((prev) => ({ ...prev, [key]: value }))
  }

  const runValidate = async () => {
    if (!canSubmit) return
    setBusy('validate')
    try {
      const next = await api.validateLoraExtraction(requestBody)
      setResult(next)
      toast(next.matched_count > 0 ? `匹配到 ${next.matched_count} 个可提取权重` : '没有匹配到可提取权重', next.matched_count > 0 ? 'success' : 'error')
    } catch (e) {
      toast(e instanceof Error ? e.message : String(e), 'error')
    } finally {
      setBusy(null)
    }
  }

  const runExtract = async () => {
    if (!canSubmit) return
    setBusy('extract')
    try {
      const next = await api.extractLoraFromFull(requestBody)
      setResult(next)
      toast(`已导出 LoRA：${next.output_path}`, 'success')
    } catch (e) {
      toast(e instanceof Error ? e.message : String(e), 'error')
    } finally {
      setBusy(null)
    }
  }

  const pickPath = (value: string) => {
    if (picker) setField(picker, value)
    setPicker(null)
  }

  return (
    <div className="fade-in flex flex-col h-full overflow-hidden">
      <PageHeader
        title="差异提取 LoRA"
        subtitle="从全量微调权重与基模权重的差值中，用 SVD 近似分解出 LoRA safetensors。"
        actions={(
          <div className="flex gap-2">
            <button className="btn btn-secondary" onClick={runValidate} disabled={!canSubmit || busy === 'validate'}>
              {busy === 'validate' ? '校验中...' : '校验匹配'}
            </button>
            <button className="btn btn-primary" onClick={runExtract} disabled={!canSubmit || busy === 'extract'}>
              {busy === 'extract' ? '提取中...' : '提取 LoRA'}
            </button>
          </div>
        )}
      />

      <div className="p-6 flex gap-4 flex-wrap xl:flex-nowrap flex-1 min-h-0 overflow-auto">
        <section className="card w-full xl:w-[460px] shrink-0 self-start" style={{ padding: 18 }}>
          <div className="flex items-baseline justify-between mb-4">
            <h3 className="m-0 text-md font-semibold">输入与输出</h3>
            <span className="caption">base + full tuned → lora</span>
          </div>

          <div className="flex flex-col gap-3">
            <PathInput label="基模 safetensors" value={form.base_path} onChange={(v) => setField('base_path', v)} onPick={() => setPicker('base_path')} />
            <PathInput label="全量微调 safetensors" value={form.tuned_path} onChange={(v) => setField('tuned_path', v)} onPick={() => setPicker('tuned_path')} />
            <PathInput label="输出 LoRA safetensors" value={form.output_path} onChange={(v) => setField('output_path', v)} onPick={() => setPicker('output_path')} />

            <div className="grid grid-cols-2 gap-3">
              <TextField label="Rank" value={form.rank} onChange={(v) => setField('rank', v)} type="number" min={1} />
              <TextField label="Alpha（空=Rank）" value={form.alpha} onChange={(v) => setField('alpha', v)} type="number" min={0.0001} step={0.5} placeholder="32" />
            </div>

            <TextField label="LoRA key prefix" value={form.prefix} onChange={(v) => setField('prefix', v)} placeholder="lora_unet" />

            <div>
              <label className="caption block mb-1.5">目标权重 pattern</label>
              <textarea
                className="input input-mono w-full resize-y text-xs"
                rows={4}
                value={form.target_pattern}
                onChange={(e) => setField('target_pattern', e.target.value)}
              />
              <div className="mt-1.5 text-2xs text-fg-tertiary font-mono">用 | 或 , 分隔 fnmatch pattern。</div>
            </div>
          </div>
        </section>

        <section className="flex-1 min-w-[360px] flex flex-col gap-4">
          <div className="card" style={{ padding: 18 }}>
            <h3 className="m-0 text-md font-semibold mb-3">注意事项</h3>
            <ul className="m-0 pl-5 text-sm text-fg-secondary space-y-1.5">
              <li>基模和全量微调权重必须来自同一架构，tensor 名称和 shape 要能对上。</li>
              <li>SVD 提取是低秩近似，rank 越低体积越小，但误差可能越高。</li>
              <li>校验会读取 safetensors 元数据和张量；大模型会占用内存和一些时间。</li>
            </ul>
          </div>

          <ResultPanel result={result} busy={busy} />
        </section>
      </div>

      {picker && (
        <PathPicker
          initialPath={form[picker]}
          dirOnly={false}
          onPick={pickPath}
          onClose={() => setPicker(null)}
        />
      )}
    </div>
  )
}

function PathInput({ label, value, onChange, onPick }: {
  label: string
  value: string
  onChange: (value: string) => void
  onPick: () => void
}) {
  return (
    <div>
      <label className="caption block mb-1.5">{label}</label>
      <div className="flex gap-2">
        <input className="input input-mono flex-1 min-w-0" value={value} onChange={(e) => onChange(e.target.value)} />
        <button className="btn btn-secondary shrink-0" type="button" onClick={onPick}>浏览</button>
      </div>
    </div>
  )
}

type TextFieldProps = {
  label: string
  value: string
  onChange: (value: string) => void
} & Omit<InputHTMLAttributes<HTMLInputElement>, 'value' | 'onChange'>

function TextField({ label, value, onChange, ...props }: TextFieldProps) {
  return (
    <div>
      <label className="caption block mb-1.5">{label}</label>
      <input className="input input-mono w-full" value={value} onChange={(e) => onChange(e.target.value)} {...props} />
    </div>
  )
}

function ResultPanel({ result, busy }: { result: ExtractLoraResult | null; busy: string | null }) {
  if (!result) {
    return (
      <div className="card flex-1 grid place-items-center min-h-[320px]" style={{ padding: 18 }}>
        <div className="text-center text-fg-tertiary text-sm">
          <div className="text-lg font-semibold text-fg-secondary mb-1">等待校验</div>
          <div>先选择两个权重文件，校验匹配后再提取。</div>
        </div>
      </div>
    )
  }

  const errorEntries = Object.entries(result.errors ?? {}).sort((a, b) => b[1] - a[1]).slice(0, 12)

  return (
    <div className="card flex-1 flex flex-col min-h-[320px]" style={{ padding: 18 }}>
      <div className="flex items-center justify-between gap-3 flex-wrap mb-4">
        <div>
          <h3 className="m-0 text-md font-semibold">提取结果</h3>
          <div className="text-xs text-fg-tertiary font-mono mt-1">{busy ? '运行中...' : result.output_path}</div>
        </div>
        <span className={result.ok ? 'badge badge-ok' : 'badge badge-err'}>
          {result.ok ? `${result.matched_count} matched` : 'no match'}
        </span>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mb-4">
        <Metric label="Rank" value={String(result.rank)} />
        <Metric label="Alpha" value={formatNumber(result.alpha)} />
        <Metric label="Zero deltas" value={String(result.zero_delta)} />
        <Metric label="Mean error" value={result.mean_error == null ? '—' : formatNumber(result.mean_error)} />
      </div>

      <div className="overflow-auto border border-subtle rounded-md">
        <table className="w-full text-sm border-collapse">
          <thead className="bg-sunken text-fg-secondary sticky top-0">
            <tr>
              <th className="text-left px-3 py-2 font-semibold">Tensor</th>
              <th className="text-left px-3 py-2 font-semibold">Shape</th>
              <th className="text-left px-3 py-2 font-semibold">Rank</th>
              <th className="text-left px-3 py-2 font-semibold">Rel. error</th>
            </tr>
          </thead>
          <tbody>
            {result.matched.map((item) => (
              <tr key={item.name} className="border-t border-subtle">
                <td className="px-3 py-2 font-mono text-xs text-fg-primary">{item.name}</td>
                <td className="px-3 py-2 font-mono text-xs text-fg-secondary">{item.shape.join('×')}</td>
                <td className="px-3 py-2 font-mono text-xs text-fg-secondary">{item.used_rank}</td>
                <td className="px-3 py-2 font-mono text-xs text-fg-secondary">{result.errors?.[item.name] == null ? '—' : formatNumber(result.errors[item.name])}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {result.truncated && <div className="mt-2 text-xs text-fg-tertiary">列表只显示前 200 个匹配权重。</div>}
      {errorEntries.length > 0 && (
        <div className="mt-3 text-xs text-fg-tertiary font-mono">
          最大误差：{errorEntries.map(([name, value]) => `${name}=${formatNumber(value)}`).join(' · ')}
        </div>
      )}
    </div>
  )
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-subtle bg-sunken px-3 py-2">
      <div className="text-2xs text-fg-tertiary uppercase tracking-wide">{label}</div>
      <div className="text-sm font-mono text-fg-primary mt-1">{value}</div>
    </div>
  )
}

function formatNumber(value: number): string {
  if (!Number.isFinite(value)) return String(value)
  if (value === 0) return '0'
  if (Math.abs(value) < 0.001) return value.toExponential(3)
  return value.toFixed(6).replace(/0+$/, '').replace(/\.$/, '')
}
