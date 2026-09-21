import { chromium } from "playwright-core";
import fs from "fs";

const CHROME = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const OUT = "C:\\Users\\abdul\\AppData\\Local\\Temp\\claude\\e--DisasterIQ\\80721aac-3eb4-49fd-a2ee-7cef1283ebe1\\scratchpad";
const URL = "http://localhost:3000";

const consoleErrors = [];
const pageErrors = [];
let pass = 0, fail = 0;
const chk = (name, ok, detail = "") => {
  console.log(`  ${ok ? "PASS" : "FAIL"}: ${name}${detail ? " — " + detail : ""}`);
  ok ? pass++ : fail++;
};

const browser = await chromium.launch({ executablePath: CHROME, headless: true });
const ctx = await browser.newContext({ acceptDownloads: true, viewport: { width: 1600, height: 1000 } });
const page = await ctx.newPage();
page.on("console", (m) => { if (m.type() === "error") consoleErrors.push(m.text()); });
page.on("pageerror", (e) => pageErrors.push(e.message));

try {
  console.log("== Load page ==");
  const resp = await page.goto(URL, { waitUntil: "networkidle", timeout: 60000 });
  chk("homepage 200", resp.status() === 200, `HTTP ${resp.status()}`);
  chk("title present", (await page.title()).length > 0, await page.title());

  // demo pairs loaded from backend into the select
  await page.waitForFunction(() => {
    const s = document.querySelector("#demo-pair");
    return s && s.options.length > 0 && s.value && !s.value.includes("offline");
  }, { timeout: 30000 });
  const pairVal = await page.$eval("#demo-pair", (s) => s.value);
  chk("demo pair loaded from backend", !!pairVal, pairVal);
  const offline = await page.locator("text=Backend offline").count();
  chk("backend NOT reported offline", offline === 0);

  await page.screenshot({ path: `${OUT}/ui_1_initial.png`, fullPage: true });

  console.log("== Click Analyze Damage (triggers real docker inference + brief) ==");
  await page.getByRole("button", { name: /Analyze Damage/i }).click();
  chk("analyze button shows Analyzing state", await page.getByRole("button", { name: /Analyzing/i }).count() > 0);

  // wait for analysis to complete (docker ~2min) then brief
  await page.waitForSelector("text=Analysis Complete", { timeout: 220000 });
  chk("analysis completed", true);
  await page.waitForSelector("text=Executive Summary", { timeout: 90000 });
  chk("brief rendered", true);

  // brief is live Fireworks
  const live = await page.locator("text=Fireworks AI").count();
  chk("brief source is live Fireworks AI", live > 0);

  // summary populated (Total Buildings not dash)
  const totalBuildings = await page.evaluate(() => {
    const el = [...document.querySelectorAll("p")].find((p) => p.textContent.trim() === "Total Buildings");
    return el ? el.parentElement.querySelector("p.text-3xl")?.textContent?.trim() : null;
  });
  chk("Total Buildings populated", totalBuildings && totalBuildings !== "—", `value=${totalBuildings}`);

  // damage overlay canvas rendered
  const canvasCount = await page.locator("canvas").count();
  chk("damage canvas present", canvasCount > 0, `${canvasCount} canvas`);

  // zone table has data rows (look for rank cells / zone rows)
  const tableText = await page.locator("body").innerText();
  chk("zone data visible (rank refs)", /rank|zone|priority/i.test(tableText));

  await page.screenshot({ path: `${OUT}/ui_2_results.png`, fullPage: true });

  console.log("== Download PDF report ==");
  const dlBtn = page.getByRole("button", { name: /Download field report/i });
  chk("download button present", await dlBtn.count() > 0);
  const [download] = await Promise.all([
    page.waitForEvent("download", { timeout: 60000 }),
    dlBtn.click(),
  ]);
  const pdfPath = `${OUT}/ui_report.pdf`;
  await download.saveAs(pdfPath);
  const head = fs.readFileSync(pdfPath).subarray(0, 5).toString("latin1");
  chk("downloaded file is a PDF", head === "%PDF-", `sig=${head}, size=${fs.statSync(pdfPath).size}`);

} catch (e) {
  console.log("  FATAL:", e.message);
  fail++;
  try { await page.screenshot({ path: `${OUT}/ui_error.png`, fullPage: true }); } catch {}
} finally {
  console.log("\n== Console errors:", consoleErrors.length, "| Page errors:", pageErrors.length);
  consoleErrors.slice(0, 10).forEach((e) => console.log("   console:", e.slice(0, 160)));
  pageErrors.slice(0, 10).forEach((e) => console.log("   pageerror:", e.slice(0, 160)));
  chk("no uncaught page errors", pageErrors.length === 0);
  console.log(`\n==================== UI RESULT: ${pass} passed, ${fail} failed ====================`);
  await browser.close();
  process.exit(fail > 0 ? 1 : 0);
}
