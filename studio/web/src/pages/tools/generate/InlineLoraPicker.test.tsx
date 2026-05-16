import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import InlineLoraPicker, { projectAbbr } from './InlineLoraPicker'
import type { ProjectLora } from './types'

const sample: ProjectLora[] = [
  {
    projectId: 1, projectTitle: 'cute_chibi',
    versionId: 11, versionLabel: 'v3', stage: 'training',
    path: '/loras/cute_chibi/v3.safetensors', createdAt: 300,
  },
  {
    projectId: 1, projectTitle: 'cute_chibi',
    versionId: 12, versionLabel: 'v2', stage: 'done',
    path: '/loras/cute_chibi/v2.safetensors', createdAt: 200,
  },
  {
    projectId: 2, projectTitle: 'noir_portrait',
    versionId: 21, versionLabel: 'v1', stage: 'done',
    path: '/loras/noir/v1.safetensors', createdAt: 100,
  },
]

describe('projectAbbr', () => {
  it('extracts first 2 alphanumerics, uppercase', () => {
    expect(projectAbbr('cute_chibi')).toBe('CU')
    expect(projectAbbr('noir_portrait')).toBe('NO')
    expect(projectAbbr('character_yui')).toBe('CH')
  })
  it('strips non-alphanumeric', () => {
    expect(projectAbbr('___test')).toBe('TE')
  })
  it('falls back to ?? when empty', () => {
    expect(projectAbbr('___')).toBe('??')
    expect(projectAbbr('')).toBe('??')
  })
})

describe('InlineLoraPicker', () => {
  const rows = () => within(screen.getByTestId('inline-lora-list'))

  function renderPicker(overrides: Partial<{
    projectLoras: ProjectLora[]
    selectedPaths: Set<string>
  }> = {}) {
    const onPick = vi.fn()
    const onClose = vi.fn()
    const onPickExternal = vi.fn()
    const utils = render(
      <InlineLoraPicker
        projectLoras={overrides.projectLoras ?? sample}
        selectedPaths={overrides.selectedPaths ?? new Set()}
        onPick={onPick}
        onClose={onClose}
        onPickExternal={onPickExternal}
      />
    )
    return { ...utils, onPick, onClose, onPickExternal }
  }

  it('shows count in header (Selected / 总)', () => {
    renderPicker()
    // 改成扁平 + Selected/总。3 个总，0 Selected
    expect(screen.getByText(/Selected 0 \/ 3/)).toBeInTheDocument()
  })

  it('shows project and version labels in each row', () => {
    renderPicker()
    expect(rows().getByText(/cute_chibi \/ v3/)).toBeInTheDocument()
    expect(rows().getByText(/cute_chibi \/ v2/)).toBeInTheDocument()
    expect(rows().getByText(/noir_portrait \/ v1/)).toBeInTheDocument()
  })

  it('filters with separate project and version dropdowns', async () => {
    const user = userEvent.setup()
    renderPicker()

    await user.selectOptions(screen.getByLabelText('Filter project'), '1')
    expect(rows().getByText(/cute_chibi \/ v3/)).toBeInTheDocument()
    expect(rows().getByText(/cute_chibi \/ v2/)).toBeInTheDocument()
    expect(rows().queryByText(/noir_portrait \/ v1/)).not.toBeInTheDocument()

    await user.selectOptions(screen.getByLabelText('Filter version'), '1:12')
    expect(rows().queryByText(/cute_chibi \/ v3/)).not.toBeInTheDocument()
    expect(rows().getByText(/cute_chibi \/ v2/)).toBeInTheDocument()
    expect(screen.getByText(/Selected 0 \/ 1/)).toBeInTheDocument()
  })

  it('shows Training badge for training-stage versions', () => {
    renderPicker()
    expect(screen.getByText('Training')).toBeInTheDocument()
  })

  it('marks already-added versions with ✓ marker; disables when no onRemove (multi-select off)', () => {
    renderPicker({
      selectedPaths: new Set(['/loras/cute_chibi/v3.safetensors']),
    })
    const addedBtn = rows().getByText(/cute_chibi \/ v3/).closest('button')!
    // 没传 onRemove → multi-select off → button disabled
    expect(addedBtn).toBeDisabled()
    expect(addedBtn.textContent).toContain('✓')
  })

  it('calls onPick with path when a version is clicked', async () => {
    const user = userEvent.setup()
    const { onPick } = renderPicker()
    const btn = rows().getByText(/noir_portrait \/ v1/).closest('button')!
    await user.click(btn)
    expect(onPick).toHaveBeenCalledWith('/loras/noir/v1.safetensors')
  })

  it('does not call onPick when clicking already-added (button disabled)', async () => {
    const user = userEvent.setup()
    const { onPick } = renderPicker({
      selectedPaths: new Set(['/loras/cute_chibi/v3.safetensors']),
    })
    const btn = rows().getByText(/cute_chibi \/ v3/).closest('button')!
    await user.click(btn)
    expect(onPick).not.toHaveBeenCalled()
  })

  it('filters by project title via search', async () => {
    const user = userEvent.setup()
    renderPicker()
    const search = screen.getByPlaceholderText('Search project / version / filename…')
    await user.type(search, 'noir')
    // cute_chibi 的版本应该消失
    expect(rows().queryByText(/cute_chibi \/ v3/)).not.toBeInTheDocument()
    expect(rows().getByText(/noir_portrait \/ v1/)).toBeInTheDocument()
    // count 更新（改成 Selected 0/总）
    expect(screen.getByText(/Selected 0 \/ 1/)).toBeInTheDocument()
  })

  it('filters by version label via search', async () => {
    const user = userEvent.setup()
    renderPicker()
    const search = screen.getByPlaceholderText('Search project / version / filename…')
    await user.type(search, 'v3')
    expect(rows().getByText(/cute_chibi \/ v3/)).toBeInTheDocument()
    expect(rows().queryByText(/cute_chibi \/ v2/)).not.toBeInTheDocument()
    expect(rows().queryByText(/noir_portrait \/ v1/)).not.toBeInTheDocument()
  })

  it('shows empty hint when no matches', async () => {
    const user = userEvent.setup()
    renderPicker()
    await user.type(screen.getByPlaceholderText('Search project / version / filename…'), 'zzznomatch')
    expect(screen.getByText(/No matching LoRA/)).toBeInTheDocument()
  })

  it('shows "no LoRAs trained" hint when projectLoras is empty', () => {
    renderPicker({ projectLoras: [] })
    expect(screen.getByText(/No trained LoRA yet/)).toBeInTheDocument()
  })

  it('triggers onClose when × is clicked', async () => {
    const user = userEvent.setup()
    const { onClose } = renderPicker()
    await user.click(screen.getByLabelText('Close picker'))
    expect(onClose).toHaveBeenCalled()
  })

  it('triggers onPickExternal when External file is clicked', async () => {
    const user = userEvent.setup()
    const { onPickExternal } = renderPicker()
    await user.click(screen.getByText('External file'))
    expect(onPickExternal).toHaveBeenCalled()
  })
})
