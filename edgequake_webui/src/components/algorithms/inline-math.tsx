'use client';

// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-ignore — katex auto-render has no type declarations
import renderMathInElement from 'katex/dist/contrib/auto-render.mjs';
import 'katex/dist/katex.min.css';
import { useEffect, useRef } from 'react';

interface InlineMathProps {
  text: string;
  className?: string;
}

/**
 * Renders text with LaTeX math expressions using KaTeX auto-render.
 * Recognises `$...$` (inline) and `$$...$$` (block) delimiters.
 */
export function InlineMath({ text, className }: InlineMathProps) {
  const ref = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    if (!ref.current) return;
    ref.current.textContent = text;
    renderMathInElement(ref.current, {
      delimiters: [
        { left: '$$', right: '$$', display: true },
        { left: '$', right: '$', display: false },
      ],
      throwOnError: false,
      strict: false,
      trust: true,
    });
  }, [text]);

  return <span ref={ref} className={className}>{text}</span>;
}
