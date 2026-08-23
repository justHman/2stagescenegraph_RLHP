import { useEffect, useState } from 'react'
import { createRoot } from 'react-dom/client'
import './styles.css'

type Choice = { name: string; probability: number }
type ObjectNode = { label: string }
type Settings = {
  top_k_relations: number
  relation_candidate_top_k: number
  object_max_count: number
  object_min_probability: number
}
type Relation = {
  id: number; subject: ObjectNode; object: ObjectNode; choices: Choice[];
  relation_score: number; triplet_score: number; relative_margin: number; uncertainty_score: number;
}
type ComparisonRow = {
  method: 'highest_probability' | 'uncertainty_priority'; rank: number; relation_id: number;
  subject: string; object: string; predicate_1: string; p1: number; predicate_2: string; p2: number;
  relation_score: number; triplet_score: number; relative_margin: number; uncertainty_score: number;
  selected_by_both: boolean;
}
type Navigator = { image_ids: string[]; active_image_id: string | null }
type Task = {
  completed: boolean; image?: string; overlay?: string; relation?: Relation;
  progress: { image?: number; images: number; relation?: number; relations?: number; labels: number };
  qa?: { id: number; subject: string; predicate: string; object: string; confidence: number; relation_score: number; triplet_score: number; uncertainty_score: number }[];
  comparison?: ComparisonRow[];
  settings?: Settings; can_undo: boolean; navigator: Navigator;
}
type Preview = {
  image: string; index: number; images: number; overlay: string;
  relation: null | { subject: ObjectNode; object: ObjectNode; predicate: string; probability: number }
}

const DEFAULT_SETTINGS: Settings = {
  top_k_relations: 3,
  relation_candidate_top_k: 15,
  object_max_count: 30,
  object_min_probability: 0.50,
}

const api = async <T,>(path: string, init?: RequestInit): Promise<T> => {
  const response = await fetch(path, init)
  if (!response.ok) throw new Error((await response.json().catch(() => ({}))).detail || 'Something went wrong.')
  return response.json()
}

function RelationGraph({ subject, object, predicates }: { subject?: ObjectNode; object?: ObjectNode; predicates: string[] }) {
  if (!subject || !object) return <p className="text-sm text-slate-500">No relation is available to display.</p>
  return <svg viewBox="0 0 460 124" className="mt-3 w-full" role="img" aria-label="Subject-predicate-object diagram">
    <defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#2563eb" /></marker></defs>
    <rect x="0" y="34" width="140" height="58" rx="14" fill="#dbeafe" stroke="#2563eb" />
    <text x="70" y="57" textAnchor="middle" className="fill-slate-900 text-[12px] font-bold">SUBJECT</text>
    <text x="70" y="78" textAnchor="middle" className="fill-slate-700 text-[14px]">{subject.label}</text>
    <line x1="142" y1="63" x2="318" y2="63" stroke="#2563eb" strokeWidth="3" markerEnd="url(#arrow)" />
    <rect x="320" y="34" width="140" height="58" rx="14" fill="#e0e7ff" stroke="#4f46e5" />
    <text x="390" y="57" textAnchor="middle" className="fill-slate-900 text-[12px] font-bold">OBJECT</text>
    <text x="390" y="78" textAnchor="middle" className="fill-slate-700 text-[14px]">{object.label}</text>
    <rect x="75" y="5" width="310" height="32" rx="11" fill="#fff7ed" stroke="#f59e0b" />
    <text x="230" y="27" textAnchor="middle" className="fill-amber-800 text-[12px] font-bold">{predicates.join(' / ')}</text>
  </svg>
}

function SettingsAndQA({ settings, setSettings, apply, busy, qa, comparison }: {
  settings: Settings
  setSettings: (settings: Settings) => void
  apply: () => void
  busy: boolean
  qa: Task['qa']
  comparison: Task['comparison']
}) {
  const invalidKM = settings.relation_candidate_top_k < settings.top_k_relations
  const numberField = (
    label: string, key: keyof Settings, min: number, max: number, step = 1,
  ) => <label className="block text-sm font-medium text-slate-700">{label}
    <input className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2" type="number"
      min={min} max={max} step={step} value={settings[key]}
      onChange={event => setSettings({...settings, [key]: Number(event.target.value)})}/>
  </label>

  return <>
    <div className="mt-4 grid gap-2 text-xs sm:grid-cols-3">
      <div className="rounded-lg border border-blue-200 bg-blue-50 p-3"><b className="block text-blue-800">1 · Object filter</b><span className="text-slate-600">confidence + maximum count</span></div>
      <div className="rounded-lg border border-indigo-200 bg-indigo-50 p-3"><b className="block text-indigo-800">2 · Top-K shortlist</b><span className="text-slate-600">highest p1 first</span></div>
      <div className="rounded-lg border border-amber-200 bg-amber-50 p-3"><b className="block text-amber-800">3 · Top-M labels</b><span className="text-slate-600">smallest relative margin</span></div>
    </div>
    <section className="mt-4 rounded-xl border border-slate-200 p-4">
      <p className="mb-3 text-sm font-bold text-slate-800">Object detection filter</p>
      <div className="grid grid-cols-2 gap-3">
        {numberField('Minimum object confidence', 'object_min_probability', 0, 1, 0.05)}
        {numberField('Maximum detected objects', 'object_max_count', 2, 100)}
      </div>
    </section>
    <section className="mt-3 rounded-xl border border-slate-200 p-4">
      <p className="mb-1 text-sm font-bold text-slate-800">Relation selection</p>
      <p className="mb-3 text-xs text-slate-500">K creates the high-probability candidate pool. M chooses the most uncertain pairs inside that pool.</p>
      <div className="grid grid-cols-2 gap-3">
        {numberField('Top-K candidates by p1', 'relation_candidate_top_k', 1, 1000)}
        {numberField('Top-M relations to label', 'top_k_relations', 1, 10)}
      </div>
      {invalidKM && <p className="mt-3 rounded-lg bg-red-50 px-3 py-2 text-xs font-semibold text-red-700">Top-K must be greater than or equal to Top-M.</p>}
    </section>
    <button disabled={busy || invalidKM} onClick={apply} className="mt-3 w-full rounded-lg bg-slate-800 px-3 py-2 text-sm font-bold text-white disabled:cursor-not-allowed disabled:opacity-50">Apply relation settings</button>
    <ul className="mt-4 space-y-2 text-sm">{qa?.map(item => <li key={item.id} className="rounded bg-slate-50 p-2">
      {item.subject} → <b>{item.predicate}</b> → {item.object}
      <span className="mt-1 block text-xs text-slate-500">p1 {Math.round(item.relation_score * 100)}% · uncertainty {Math.round(item.uncertainty_score * 100)}%</span>
    </li>)}</ul>
    <div className="mt-5 overflow-x-auto">
      <p className="mb-2 text-sm font-bold text-slate-700">Selection comparison</p>
      <table className="w-full text-left text-xs">
        <thead><tr className="border-b text-slate-500"><th className="py-2 pr-2">Method</th><th className="pr-2">#</th><th className="pr-2">Relation</th><th className="pr-2">Top predicates</th><th className="pr-2">p1</th><th>Uncertainty</th></tr></thead>
        <tbody>{comparison?.map(row => <tr key={`${row.method}-${row.rank}`} className="border-b border-slate-100 align-top">
          <td className="py-2 pr-2 font-semibold">{row.method === 'highest_probability' ? 'Top probability' : 'Uncertainty'}</td>
          <td className="pr-2">{row.rank}</td><td className="pr-2">{row.subject} → {row.object}</td>
          <td className="pr-2">{row.predicate_1} / {row.predicate_2}</td>
          <td className="pr-2">{(row.p1 * 100).toFixed(1)}%</td><td>{(row.uncertainty_score * 100).toFixed(1)}%</td>
        </tr>)}</tbody>
      </table>
    </div>
  </>
}

function App() {
  const [files, setFiles] = useState<File[]>([])
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [task, setTask] = useState<Task | null>(null)
  const [preview, setPreview] = useState<Preview | null>(null)
  const [previewId, setPreviewId] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const start = async () => {
    if (!files.length) return setError('Choose a ZIP file or at least one image.')
    if (settings.relation_candidate_top_k < settings.top_k_relations) return setError('Top-K candidates must be greater than or equal to Top-M labels.')
    setBusy(true); setError('')
    try {
      const form = new FormData(); files.forEach(file => form.append('files', file))
      const result = await api<{ id: string; task: Task }>(`/api/sessions?top_k_relations=${settings.top_k_relations}&relation_candidate_top_k=${settings.relation_candidate_top_k}`, { method: 'POST', body: form })
      setSessionId(result.id); setTask(result.task); setPreview(null)
      if (result.task.settings) setSettings(result.task.settings)
    } catch (e) { setError(e instanceof Error ? e.message : 'Unable to start the batch.') } finally { setBusy(false) }
  }
  const action = async (choice: string) => {
    if (!sessionId || !task?.relation || preview) return
    setBusy(true); setError('')
    try { setTask(await api<Task>(`/api/sessions/${sessionId}/responses`, { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({image_id: task.image, relation_id: task.relation.id, presented_predicates: task.relation.choices, choice}) })) }
    catch (e) { setError(e instanceof Error ? e.message : 'Unable to save the label.') } finally { setBusy(false) }
  }
  const undo = async () => {
    if (!sessionId || !task?.can_undo || busy) return
    setBusy(true); setError('')
    try { setPreview(null); setTask(await api<Task>(`/api/sessions/${sessionId}/undo`, {method:'POST'})) }
    catch (e) { setError(e instanceof Error ? e.message : 'Unable to undo the last action.') } finally { setBusy(false) }
  }
  const openPreview = async (imageId: string) => {
    if (!sessionId || !imageId || busy) return
    setBusy(true); setError('')
    try { setPreview(await api<Preview>(`/api/sessions/${sessionId}/preview?image_id=${encodeURIComponent(imageId)}`)); setPreviewId(imageId) }
    catch (e) { setError(e instanceof Error ? e.message : 'Unable to open the image preview.') } finally { setBusy(false) }
  }
  const applySettings = async () => {
    if (!sessionId || busy) return
    setBusy(true); setError('')
    try {
      const updated = await api<Task>(`/api/sessions/${sessionId}/settings`, {
        method:'PATCH', headers:{'Content-Type':'application/json'}, body:JSON.stringify(settings),
      })
      setTask(updated); setPreview(null)
      if (updated.settings) setSettings(updated.settings)
    } catch (e) { setError(e instanceof Error ? e.message : 'Unable to update settings.') }
    finally { setBusy(false) }
  }
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if ((event.target as HTMLElement)?.matches('input, textarea, select') || busy) return
      const key = event.key.toLowerCase()
      if ((event.ctrlKey || event.metaKey) && key === 'z') { event.preventDefault(); void undo(); return }
      if (key === 'z') { event.preventDefault(); void undo(); return }
      const keys: Record<string, string> = {'1':'first','2':'second','0':'neutral','x':'wrong','s':'skip'}
      if (keys[key] && !preview) { event.preventDefault(); void action(keys[key]) }
    }
    window.addEventListener('keydown', onKey); return () => window.removeEventListener('keydown', onKey)
  }, [task, sessionId, busy, preview])

  if (!sessionId) return <main className="mx-auto flex min-h-screen max-w-2xl items-center p-6"><section className="w-full rounded-3xl bg-white p-8 shadow-xl ring-1 ring-slate-200"><p className="text-sm font-semibold text-blue-700">SCENE GRAPH LABELER</p><h1 className="mt-2 text-3xl font-bold text-slate-950">Start a batch</h1><p className="mt-2 text-slate-600">Choose a ZIP file or multiple images. Every selection is saved automatically.</p><label className="mt-7 block rounded-2xl border-2 border-dashed border-blue-300 bg-blue-50 p-8 text-center text-slate-700"><input className="sr-only" type="file" multiple accept=".zip,.jpg,.jpeg,.png,.webp" onChange={e => setFiles(Array.from(e.target.files || []))}/><span className="font-semibold text-blue-700">Choose ZIP or images</span><br/><span className="text-sm">{files.length ? `${files.length} file${files.length === 1 ? '' : 's'} selected` : 'JPG, PNG, WebP, or ZIP'}</span></label><div className="mt-6 grid grid-cols-2 gap-3 rounded-xl border border-slate-200 p-4"><label className="block font-semibold text-slate-800">Top-K candidates by p1<input className="mt-2 w-full rounded-lg border border-slate-300 px-3 py-2" type="number" min="1" max="1000" value={settings.relation_candidate_top_k} onChange={e => setSettings({...settings, relation_candidate_top_k:Number(e.target.value)})}/></label><label className="block font-semibold text-slate-800">Top-M relations to label: <span className="text-blue-700">{settings.top_k_relations}</span><input className="mt-2 block w-full accent-blue-600" type="range" min="1" max="10" value={settings.top_k_relations} onChange={e => setSettings({...settings, top_k_relations:Number(e.target.value)})}/></label></div><p className="mt-2 text-xs text-slate-500">Object filter → Top-K by p1 → Top-M by smallest relative margin. K must be at least M.</p>{settings.relation_candidate_top_k < settings.top_k_relations && <p className="mt-3 rounded-lg bg-red-50 p-3 text-sm font-semibold text-red-700">Top-K must be greater than or equal to Top-M.</p>}{error && <p className="mt-4 rounded-lg bg-red-50 p-3 text-red-700">{error}</p>}<button disabled={busy || settings.relation_candidate_top_k < settings.top_k_relations} onClick={start} className="mt-6 w-full rounded-xl bg-blue-700 px-5 py-3 font-bold text-white hover:bg-blue-800 disabled:cursor-not-allowed disabled:opacity-50">{busy ? 'Preparing…' : 'Start labeling'}</button></section></main>

  const ids = task?.navigator.image_ids || []
  const visibleImage = preview?.image || task?.image
  const visibleOverlay = preview?.overlay || task?.overlay
  const visibleRelation = preview?.relation
  const currentPosition = preview ? preview.index - 1 : Math.max(0, ids.indexOf(task?.navigator.active_image_id || ''))
  const graphSubject = preview ? visibleRelation?.subject : task?.relation?.subject
  const graphObject = preview ? visibleRelation?.object : task?.relation?.object
  const graphPredicates = preview ? (visibleRelation ? [`${visibleRelation.predicate} ${(visibleRelation.probability * 100).toFixed(0)}%`] : []) : (task?.relation?.choices.map(choice => `${choice.name} ${(choice.probability * 100).toFixed(0)}%`) || [])

  return <main className="min-h-screen bg-slate-100 p-4 lg:p-7"><header className="mx-auto mb-4 flex max-w-[1500px] items-center justify-between gap-3"><div><p className="font-bold text-slate-900">Scene Graph Labeler</p><p className="text-sm text-slate-600">{task?.progress.labels || 0} labels saved</p></div><div className="flex gap-2"><button disabled={!task?.can_undo || busy} onClick={() => void undo()} className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm font-semibold text-slate-700 disabled:cursor-not-allowed disabled:opacity-40">↶ Undo · Ctrl+Z</button><a className="rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-semibold text-slate-700" href={`/api/sessions/${sessionId}/export`}>Download CSV</a></div></header><div className="mx-auto mb-4 flex max-w-[1500px] flex-wrap items-center gap-2 rounded-xl bg-white p-3 shadow-sm ring-1 ring-slate-200"><span className="text-sm font-semibold text-slate-700">Browse samples</span><button disabled={busy || currentPosition <= 0} onClick={() => void openPreview(ids[currentPosition - 1])} className="nav-button">← Previous</button><button disabled={busy || currentPosition >= ids.length - 1} onClick={() => void openPreview(ids[currentPosition + 1])} className="nav-button">Next →</button><input list="image-ids" value={previewId} placeholder={visibleImage || 'Image ID'} onChange={e => setPreviewId(e.target.value)} onKeyDown={e => { if (e.key === 'Enter') void openPreview(previewId) }} className="min-w-52 rounded-lg border border-slate-300 px-3 py-2 text-sm"/><datalist id="image-ids">{ids.map(id => <option key={id} value={id}/>)}</datalist><button disabled={busy || !previewId} onClick={() => void openPreview(previewId)} className="nav-button">Open ID</button>{preview && <button disabled={busy} onClick={() => { setPreview(null); setPreviewId('') }} className="ml-auto rounded-lg bg-blue-700 px-3 py-2 text-sm font-semibold text-white">Return to labeling</button>}</div>{preview && <p className="mx-auto mb-3 max-w-[1500px] rounded-lg bg-amber-50 px-4 py-2 text-sm text-amber-800">Previewing <b>{preview.image}</b>. Predicate choices are disabled and autosave progress is unchanged.</p>}{task?.completed && !preview ? <section className="mx-auto max-w-xl rounded-2xl bg-white p-10 text-center shadow"><h1 className="text-2xl font-bold">Batch complete</h1><p className="mt-2 text-slate-600">You labeled {task.progress.labels} relations. Browse samples above to review them.</p></section> : <div className="mx-auto grid max-w-[1500px] gap-5 lg:grid-cols-[minmax(0,7fr)_minmax(320px,3fr)]"><section className="rounded-2xl bg-slate-950 p-3 shadow"><img className="max-h-[76vh] w-full rounded-xl object-contain" src={visibleOverlay} alt={`Image ${visibleImage || ''}`}/></section><aside className="rounded-2xl bg-white p-6 shadow ring-1 ring-slate-200"><p className="text-sm font-semibold text-blue-700">{preview ? `Previewing image ${preview.index}/${preview.images}` : `Image ${task?.progress.image}/${task?.progress.images} · Relation ${task?.progress.relation}/${task?.progress.relations}`}</p><h1 className="mt-4 text-2xl font-bold text-slate-950">{graphSubject?.label} <span className="text-slate-400">→</span> {graphObject?.label}</h1><section className="mt-4 rounded-xl border border-slate-200 bg-slate-50 p-3"><p className="text-xs font-bold tracking-wide text-slate-600">NODE · PREDICATE · NODE</p><RelationGraph subject={graphSubject} object={graphObject} predicates={graphPredicates}/></section>{!preview && <><p className="mt-5 text-slate-600">Which relation is more accurate?</p><p className="mt-1 text-xs text-slate-500">Uncertainty {Math.round((task?.relation?.uncertainty_score || 0) * 100)}% · p1 {Math.round((task?.relation?.relation_score || 0) * 100)}%</p><div className="mt-4 grid gap-3"><button disabled={busy} onClick={() => action('first')} className="choice bg-blue-700 hover:bg-blue-800">1 · {task?.relation?.choices[0]?.name}</button><button disabled={busy} onClick={() => action('second')} className="choice bg-indigo-600 hover:bg-indigo-700">2 · {task?.relation?.choices[1]?.name}</button><button disabled={busy} onClick={() => action('neutral')} className="choice bg-amber-500 hover:bg-amber-600">0 · Unsure / equivalent</button><button disabled={busy} onClick={() => action('wrong')} className="choice bg-red-600 hover:bg-red-700">X · No relation</button><button disabled={!task?.can_undo || busy} onClick={() => void undo()} className="rounded-xl border border-slate-300 bg-white px-4 py-3 font-bold text-slate-800 hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-40">↶ Undo · Ctrl+Z</button></div></>} {error && <p className="mt-4 text-sm text-red-700">{error}</p>} {!preview && <details className="mt-6 border-t pt-4"><summary className="cursor-pointer font-semibold text-slate-700">Settings & QA</summary><SettingsAndQA settings={settings} setSettings={setSettings} apply={() => void applySettings()} busy={busy} qa={task?.qa} comparison={task?.comparison}/></details>}</aside></div>}</main>
}

createRoot(document.getElementById('root')!).render(<App />)
