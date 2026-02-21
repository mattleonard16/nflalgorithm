import { chromium } from "playwright-core";
import { writeFileSync, mkdirSync } from "node:fs";
import { parseArgs } from "node:util";
import { dirname } from "node:path";

const MAX_TEXT = 8000;
const SPA_THRESHOLD = 200;

interface LinkInfo {
  text: string;
  href: string;
}
interface SourceResult {
  url: string;
  title: string;
  text: string;
  links: LinkInfo[];
  screenshotPath?: string;
}
interface ResearchOutput {
  question: string;
  date: string;
  sources: SourceResult[];
}

async function harvestSource(
  context: Awaited<ReturnType<typeof chromium.connectOverCDP>>["contexts"][0],
  url: string,
  index: number,
  outputDir: string
): Promise<SourceResult> {
  const page = await context.newPage();
  try {
    await page.setViewportSize({ width: 1440, height: 900 });
    await page.goto(url, { waitUntil: "networkidle", timeout: 15000 });

    const title = await page.title();
    const text = await page.evaluate(() => document.body.innerText);

    let screenshotPath: string | undefined;
    if (text.length < SPA_THRESHOLD) {
      screenshotPath = `${outputDir}/source-${index}.png`;
      await page.screenshot({ path: screenshotPath, fullPage: true });
    }

    const links: LinkInfo[] = await page.evaluate(() =>
      Array.from(document.querySelectorAll("a[href]"))
        .slice(0, 50)
        .map((a) => ({
          text: (a as HTMLAnchorElement).innerText.trim().slice(0, 100),
          href: (a as HTMLAnchorElement).href,
        }))
        .filter((l) => l.text && l.href.startsWith("http"))
    );

    return { url, title, text: text.slice(0, MAX_TEXT), links, screenshotPath };
  } finally {
    await page.close();
  }
}

function buildNoteShell(output: ResearchOutput): string {
  const sourceList = output.sources.map((s) => `  - ${s.url}`).join("\n");
  return `---
question: "${output.question}"
date: ${output.date}
sources:
${sourceList}
tags: []
---

## Decision

<!-- One paragraph. What to do or avoid. -->

## Rationale

<!-- 2-4 bullets, each citing a specific source. -->
${output.sources.map((s) => `- [${s.title}](${s.url}) — `).join("\n")}

## Commands / Code Patterns

\`\`\`bash
# copy-paste ready
\`\`\`

## Links

${output.sources.map((s) => `- [${s.title}](${s.url})`).join("\n")}
`;
}

async function main() {
  const { values } = parseArgs({
    options: {
      question: { type: "string" },
      urls: { type: "string" },
      output: { type: "string" },
      "dry-run": { type: "boolean", short: "n" },
    },
  });

  if (!values.question || !values.urls) {
    console.error(
      "Usage: --question '...' --urls 'url1,url2' [--output path.md] [--dry-run]"
    );
    process.exit(1);
  }

  const urls = values.urls
    .split(",")
    .map((u) => u.trim())
    .slice(0, 5);

  if (values["dry-run"]) {
    const output: ResearchOutput = {
      question: values.question,
      date: new Date().toISOString().slice(0, 10),
      sources: urls.map((url) => ({
        url,
        title: "(dry-run)",
        text: "",
        links: [],
      })),
    };
    console.log(JSON.stringify(output, null, 2));
    if (values.output) {
      mkdirSync(dirname(values.output), { recursive: true });
      writeFileSync(values.output, buildNoteShell(output));
      console.error(`\nNote shell written to: ${values.output}`);
    }
    return;
  }

  // Connect to running dev-browser server
  let serverInfo: { wsEndpoint: string };
  try {
    const resp = await fetch("http://127.0.0.1:9222", {
      signal: AbortSignal.timeout(5000),
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    serverInfo = (await resp.json()) as { wsEndpoint: string };
    if (!serverInfo?.wsEndpoint) throw new Error("No wsEndpoint in response");
  } catch (err) {
    const msg =
      err instanceof Error ? err.message : String(err);
    console.error(
      "Cannot connect to dev-browser. Start it first:\n" +
        "  ~/.claude/plugins/cache/n-skills/dev-browser/1.0.0/skills/dev-browser/server.sh --headless &\n" +
        "  # Wait for 'Ready' message, then run this script again.\n\n" +
        `Error: ${msg}`
    );
    process.exit(1);
  }
  const browser = await chromium.connectOverCDP(serverInfo.wsEndpoint);
  const context = browser.contexts()[0] ?? (await browser.newContext());

  const outputDir = values.output ? dirname(values.output) : "docs/research";
  mkdirSync(outputDir, { recursive: true });

  const sources: SourceResult[] = [];
  for (const [i, url] of urls.entries()) {
    try {
      console.error(`[${i + 1}/${urls.length}] Fetching: ${url}`);
      const result = await harvestSource(context, url, i, outputDir);
      sources.push(result);
    } catch (err) {
      console.error(
        `[${i + 1}/${urls.length}] FAILED: ${(err as Error).message}`
      );
    }
  }

  browser.close().catch(() => {});

  const output: ResearchOutput = {
    question: values.question,
    date: new Date().toISOString().slice(0, 10),
    sources,
  };

  // JSON to stdout for agent/user to synthesize
  console.log(JSON.stringify(output, null, 2));

  // Markdown shell to --output
  if (values.output) {
    writeFileSync(values.output, buildNoteShell(output));
    console.error(`\nNote shell written to: ${values.output}`);
  }
}

main().catch((err) => {
  console.error("Fatal:", err.message);
  process.exit(1);
});
