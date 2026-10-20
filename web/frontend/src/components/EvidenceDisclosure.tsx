import { useId, useState } from 'react'
import type { ReactNode } from 'react'

/** Accessible progressive disclosure that does not mount dense evidence until requested. */
export default function EvidenceDisclosure({
  title,
  summary,
  children,
}: {
  title: string
  summary: string
  children: ReactNode
}) {
  const [open, setOpen] = useState(false)
  const id = useId()
  return (
    <section className={open ? 'evidence-disclosure is-open' : 'evidence-disclosure'}>
      <button
        type="button"
        className="evidence-disclosure-toggle"
        aria-expanded={open}
        aria-controls={id}
        onClick={() => setOpen((value) => !value)}
      >
        <span>{title}</span>
        <span>{summary}</span>
      </button>
      {open && <div id={id} className="evidence-disclosure-body">{children}</div>}
    </section>
  )
}
