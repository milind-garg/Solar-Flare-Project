# UI redesign plan

## Goal

Make the Aditya-L1 project feel like a credible scientific instrument and research portal: calm, precise, readable, and grounded in its real data. The landing page should explain the mission and earn trust; the dashboard should help an operator inspect observations and forecasts quickly. Keep the existing science, routes, controls, and chart data intact while redesigning their presentation.

## What makes the current UI feel generated

- Both surfaces lean on the same near-black palette, blue/cyan accents, and broadly applied serif typography. Dashboard labels, dropdowns, chart titles, and body copy all inherit a display-like serif face.
- The landing page layers a solar photo, vignette, grid, canvas particles, telemetry strip, parallax, scroll progress, glowing gradients, and multiple HUD decorations. Many elements compete for attention before the science is understood.
- The hero presents a simulated solar viewer with crosshairs and active-region annotations. If the imagery or coordinates are illustrative rather than live observation data, the treatment can imply a live instrument feed that the page does not establish.
- Cards repeat rounded corners, shadows, translucent fills, and accent colors across metrics, controls, and content. The interface lacks a strong distinction between navigation, scientific evidence, and decoration.
- The dashboard uses card-like KPI tiles and a dense strip of controls, while the tab names and chart-heavy body do not clearly establish a primary operator task.
- Current styles describe themselves as “Aerospace Slate” and “Cinematic Space Observatory”; the visual concept is more generic space-tech styling than a distinct identity for this mission and dataset.

## Recommended direction: an observatory field log

Use the visual language of a well-edited mission operations report: an ink-blue-black base, warm paper-like text, restrained instrument colors, hairline separators, clear typography, and plots treated as the main visual evidence. Use real instrument names, units, timestamps, caveats, and provenance as the distinguishing design details. Prefer precise editorial hierarchy over sci-fi framing.

This keeps a dark interface suitable for plots while removing unnecessary atmosphere. The solar image can remain, but should be presented as a clearly captioned scientific image with source/date and an explicit “illustrative” label wherever it is not a current instrument observation.

### Visual system

| Element | Direction |
| --- | --- |
| Base | Deep ink `#101820`; raised regions `#17232B`; avoid translucent glass surfaces. |
| Text | Warm near-white `#F1F0E8`; secondary slate `#A7B0B2`; muted labels `#78858A`. Maintain readable contrast. |
| Accent | Solar ochre `#E3A64B` for solar/flare emphasis; pale blue `#79B8C8` for SXR; violet `#B29AE0` for HXR only where channel identity helps. Reserve red for critical alerts and green for confirmed healthy status. |
| Type | One neutral sans-serif for interface and long-form copy; a tabular mono for UTC, values, and units. Use size/weight/spacing rather than a second display serif. Bundle fonts or provide system fallbacks. |
| Edges | Mostly square or subtly rounded (2–4 px); 1 px dividers; no ambient card shadows. Use a larger radius only for controls where it improves usability. |
| Layout | Left-aligned content, consistent 8 px spacing rhythm, a readable max width on the landing page, and full-width chart workspace in the dashboard. Separate sections with whitespace and rules, not nested cards. |
| Motion | Remove parallax and ambient particles. Keep only short, functional transitions; honor `prefers-reduced-motion`. No animated metric counts for scientific values. |

## Page plan

### 1. Landing page (`index.html`)

1. **Header:** Replace the telemetry ticker with a compact mission masthead: project/mission identity, a few meaningful links (Instruments, Method, Validation, Catalog), and one clear “Open dashboard” action. Keep only live UTC if it is genuinely useful; label its timezone.
2. **Hero:** Use a two-column editorial introduction. Left: a concise headline such as “Solar flare monitoring from Aditya-L1,” a one-sentence explanation of nowcasting and the 30-minute forecast horizon, and two actions (“Explore observations”, “How it works”). Right: one real solar image or a real light-curve crop, with instrument, observation date, source, and image status in a quiet caption. Avoid fabricated HUD markings.
3. **Evidence row:** Replace colorful counter cards with 3–4 aligned facts in a single ruled row: held-out AUC and split definition, lead-time statistic and its population, catalog event count and coverage, and model/data provenance. Keep a short methodology note adjacent to each statistic; do not imply performance guarantees.
4. **Instrument section:** Present SoLEXS and HEL1OS side by side as a comparison of energy range, role, and data product. Use a small real schematic or annotated spectrum only if it clarifies the instruments.
5. **Forecast method:** Turn the current feature matrix into a legible process explanation: observations → preprocessing/features → nowcast detector + forecast model → alert output. Keep the Neupert effect explanation attached to the HXR/SXR relationship it describes.
6. **Validation and catalog:** Lead with model evaluation and sample limitations; then offer the catalog as a searchable or browsable data destination. Distinguish measured observations from model predictions.
7. **Footer:** Include repository/source, instrument/data acknowledgements, model limitations, and version/update date.

### 2. Operations dashboard (`dashboard/app.py` and `dashboard/assets/theme.css`)

1. **Operator header:** A compact title, selected date/instrument, and data/model status with plain-language states. Status must distinguish “data loaded,” “model available,” and “live feed” rather than suggesting real-time operation when the view is using a static file.
2. **Control area:** Group date, instrument, benchmark event, and threshold into one clearly labeled toolbar. Keep advanced analytics options behind a disclosure if they are infrequently used. Give controls consistent heights and sans-serif labels; keep mono only for dates and numeric values.
3. **Primary observation:** Make the soft X-ray curve the dominant workspace. Put its title, selected instrument, units, time range, and data source directly above the plot. Keep gridlines faint and sparse; use a solid background and consistent axis styles.
4. **Cross-instrument context:** Place HXR and hardness ratio below or beside the main curve with shared time range and coordinated event markers. Use color consistently for SXR/HXR and label flare-class encoding.
5. **Forecast view:** Display the 30-minute probability trace, threshold, current probability, and latest forecast time together. Explain threshold status in text and visually distinguish an observed flare from a predicted event.
6. **Catalog view:** Prioritize a readable event table with useful sorting/filtering, aligned numeric fields, and clear empty/loading/error states. Make a selected row visibly link to its event context.
7. **Alerts and states:** Reserve red for critical states; include the reason, time, and affected instrument. Provide concise loading, missing-data, missing-model, and offline copy. Never rely on color alone.

## Interaction and accessibility

- Preserve current deep links, routes, benchmark presets, tabs, date/instrument selection, threshold behavior, and graph interactions.
- Ensure keyboard focus is visible, all controls have programmatic labels, tab navigation works, and targets remain usable at narrow widths.
- Use minimum 4.5:1 contrast for normal text where feasible and do not encode flare classes or statuses by hue alone.
- On mobile, stack the masthead and hero, collapse the dashboard toolbar into a wrapping or disclosure layout, and keep charts horizontally readable without clipping axis labels.
- Add explicit source, units, time zone, observation/model status, and caveats to scientific visuals.
- Prefer reduced motion by default; provide no decorative animation that competes with graph reading.

## Implementation sequence

1. Agree on the observatory field-log direction and confirm the source/status labels for landing-page imagery and dashboard data.
2. Establish shared color/type/spacing tokens and remove the universal serif defaults. Keep Plotly colors and Dash controls tied to the same tokens.
3. Redesign the dashboard shell and primary light curve first; this is the core working surface and gives the visual system a practical test.
4. Update forecast, catalog, alert, and empty/loading states using the same hierarchy.
5. Simplify the landing page from top to bottom, removing effects that do not explain mission/data, then align it with the new tokens.
6. Review desktop and mobile layouts, keyboard focus, contrast, graph labels, and all existing route/control behaviors before shipping.

## Success criteria

- A first-time visitor can explain what the system does and what its evidence represents after scanning the hero and evidence row.
- An operator can identify date, instrument, data/model availability, and the primary curve without searching through decorative chrome.
- SXR, HXR, predicted probability, flare events, and alert severity have consistent, distinguishable encodings across pages.
- Each scientific number or plot has the necessary unit, period/split, source, and observed-versus-predicted context.
- The same interface remains understandable with motion disabled, without color perception, and at mobile widths.
- The current application routes, selection controls, benchmark links, and analytical meaning remain intact.

## Out of scope for this plan

This document proposes a visual and information-architecture direction; it does not change the application, scientific claims, data pipeline, or model behavior.
