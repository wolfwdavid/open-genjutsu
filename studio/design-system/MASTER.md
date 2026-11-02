# Render Studio — design system (master)

Generated with the UI/UX Pro Max design-system search for "video editing render dashboard,
professional tool, dark, content-dense" and adjusted where noted. Dark only.

## Style: Dark editing suite (OLED-dark, flat, dense)
Reference feel: DaVinci Resolve / Premiere panels. Low white emission, visible 1 px borders on
every surface (no borderless cards on dark), minimal glow, no gradients on data.

## Colors (tokens in `ui/src/app.css` `@theme`)
| Role | Token | Hex | Note |
|---|---|---|---|
| Background | `--color-bg` | `#0b0d10` | neutral, not blue-black: the footage is orange and must read true |
| Surface / panel | `--color-surface` | `#13161b` | |
| Raised (cards, inputs) | `--color-raised` | `#1a1e25` | |
| Border | `--color-line` / `-strong` | `#262b34` / `#343b47` | visible on every surface |
| Text | `--color-fg` | `#e8eaed` | 15:1 on bg |
| Text muted | `--color-fg-muted` | `#9aa3ad` | 6.9:1 |
| Text dim (labels) | `--color-fg-dim` | `#6b7480` | 4.6:1, labels ≥ 11 px uppercase |
| Accent (primary, progress) | `--color-accent` | `#f59e0b` | the set's orange instead of the search's pink: ties the UI to the clip; 9.3:1 on bg; text on accent is `#1a1200` |
| Info / timeline | `--color-info` | `#5b9dff` | |
| OK / Warn / Danger | `--color-ok` `--color-warn` `--color-danger` | `#34d399` `#fbbf24` `#f87171` | always paired with an icon or word, never color alone |

## Typography
Fira Sans (body 15 px / 1.5, headings 600) + Fira Code for every number, timecode, path and log
(`font-variant-numeric: tabular-nums` so counters never jiggle). Google Fonts with system fallbacks.

## Components
- Buttons: 40 px min height, 1 px border, 150 ms color transitions, press scale .985; one primary
  (orange) per screen; destructive = red border, confirms before replacing a reference.
- Pills: 24 px, semantic variants (ok / warn / danger / info / accent).
- Progress: segmented window bar (done = accent, running = striped accent 40 %, pending = line);
  secondary thin blue bar for the sampler step.
- Focus: 2 px accent outline on every control; `prefers-reduced-motion` disables stripes/pulses.
- Icons: Lucide only, 16–20 px, stroke 2. No emoji.

## Layout
220 px left rail (icons + labels + key hints) on ≥ 768 px, top bar below; content max 6xl/7xl,
16–24 px gutters, 4/8 px spacing scale. Shot review: strip (flex-weighted by duration) → compare
pane + 300 px decision rail. Works at 375 px (rail collapses to a scrollable top bar).

## Anti-patterns avoided
Light mode default; blue-tinted blacks under orange footage; color-only status; hover-only
affordances; emoji icons; borderless dark cards; animating width/height (progress uses width
transition only on the thin step bar, 500 ms).
