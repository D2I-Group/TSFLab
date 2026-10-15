import Image from "next/image";

/**
 * Diaugeia.AI wordmark — the light-beam mark in a dark rounded tile, paired
 * with the logotype. Matches the official Diaugeia.AI site header.
 */
export function Logo({ className = "" }: { className?: string }) {
  return (
    <span className={`inline-flex items-center gap-2.5 ${className}`}>
      <span className="relative inline-block h-8 w-8 shrink-0 overflow-hidden rounded-lg bg-black ring-1 ring-border-strong">
        <Image src="/diaugeia.png" alt="Diaugeia.AI logo" fill sizes="48px" className="object-cover" priority />
      </span>
      <span className="font-serif text-[1.35rem] leading-none tracking-[-0.01em] text-ink">
        Diaugeia<span className="text-accent">.AI</span>
      </span>
    </span>
  );
}
