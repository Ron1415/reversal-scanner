/* Trade journal storage shared by the dashboard and the tracker.
 *
 * Trades live in trades.json in the GitHub repository, so they are kept
 * permanently, work from any device, and are read by analysis.py on each scan.
 * Writing needs a fine-grained token with "Contents: read and write" (and
 * "Actions: read and write" for the refresh buttons) on this repository. The
 * token is stored only in this browser.
 */
const Store = (() => {
  const CFG = window.CFG || {};
  const KEY = "gh_token";
  const ls = {get: k => { try { return localStorage.getItem(k); } catch (e) { return null; } },
              set: (k, v) => { try { localStorage.setItem(k, v); } catch (e) {} }};
  const api = path => `https://api.github.com/repos/${CFG.repo}/${path}`;
  const enc = s => btoa(unescape(encodeURIComponent(s)));
  const dec = s => decodeURIComponent(escape(atob(s.replace(/\n/g, ""))));
  let cache = {trades: [], sha: null};

  function token() { return ls.get(KEY); }
  function quoteKey() { return ls.get("fh_key"); }

  /* Live price. Yahoo Finance blocks requests from web pages, so the page uses
   * Finnhub's free real-time US quote service (free key from finnhub.io).
   * Returns {price, time, source} or null. */
  async function quote(ticker) {
    const k = quoteKey();
    if (!k) return null;
    try {
      const r = await fetch(`https://finnhub.io/api/v1/quote?symbol=${encodeURIComponent(ticker)}&token=${encodeURIComponent(k)}`,
                            {cache: "no-store"});
      if (!r.ok) return null;
      const j = await r.json();
      if (!(j.c > 0)) return null;
      return {price: Math.round(j.c * 100) / 100, time: new Date((j.t || Date.now() / 1000) * 1000).toISOString(),
              source: "live"};
    } catch (e) { return null; }
  }
  function setToken(t) { ls.set(KEY, (t || "").trim()); }
  function canWrite() { return !!(CFG.repo && token()); }
  function headers() { return {Authorization: `Bearer ${token()}`, Accept: "application/vnd.github+json"}; }

  async function load() {
    if (canWrite()) {
      const r = await fetch(api(`contents/trades.json?ref=${CFG.branch || "main"}&t=${Date.now()}`),
                            {headers: headers(), cache: "no-store"});
      if (r.status === 404) { cache = {trades: [], sha: null}; return cache.trades; }
      if (!r.ok) throw new Error(`GitHub read failed (${r.status}). Check the token.`);
      const j = await r.json();
      cache = {trades: JSON.parse(dec(j.content) || "[]"), sha: j.sha};
      return cache.trades;
    }
    try {   // read-only fallback: the copy published with the last scan
      const r = await fetch(`data/trades.json?t=${Date.now()}`, {cache: "no-store"});
      if (r.ok) cache.trades = await r.json();
    } catch (e) {}
    return cache.trades;
  }

  /* Apply `mutate(trades)` and save; retries if someone else saved first. */
  async function update(mutate, message) {
    if (!canWrite()) throw new Error("Add your GitHub token (settings link) to save trades.");
    for (let attempt = 0; attempt < 3; attempt++) {
      await load();
      const trades = JSON.parse(JSON.stringify(cache.trades));
      mutate(trades);
      const body = {message, content: enc(JSON.stringify(trades, null, 1)), branch: CFG.branch || "main"};
      if (cache.sha) body.sha = cache.sha;
      const r = await fetch(api("contents/trades.json"), {method: "PUT", headers: headers(), body: JSON.stringify(body)});
      if (r.ok) { cache = {trades, sha: (await r.json()).content.sha}; return trades; }
      if (r.status !== 409 && r.status !== 422) throw new Error(`GitHub save failed (${r.status}). The token needs "Contents: read and write".`);
    }
    throw new Error("Could not save after 3 attempts; please try again.");
  }

  function returnPct(t, price) {
    if (price == null || !t.entry_price) return null;
    const r = t.side === "short" ? (t.entry_price - price) / t.entry_price : (price - t.entry_price) / t.entry_price;
    return Math.round(r * 10000) / 100;
  }

  /* Read any JSON file from the repository (e.g. checks/latest.json). */
  async function readRepoJson(path) {
    if (!canWrite()) return null;
    const r = await fetch(api(`contents/${path}?ref=${CFG.branch || "main"}&t=${Date.now()}`), {headers: headers(), cache: "no-store"});
    if (!r.ok) return null;
    return JSON.parse(dec((await r.json()).content));
  }

  async function dispatch(inputs) {
    const r = await fetch(api(`actions/workflows/${CFG.workflow || "scan.yml"}/dispatches`), {
      method: "POST", headers: headers(), body: JSON.stringify({ref: CFG.branch || "main", inputs})});
    if (r.status !== 204) throw new Error(`GitHub refused the request (${r.status}). Check the token.`);
  }

  const fmtTime = iso => iso ? new Date(iso).toLocaleString("en-US", {timeZone: "America/New_York",
    month: "short", day: "numeric", hour: "numeric", minute: "2-digit"}) + " ET" : "";

  /* Small settings panel: <div id="tokenPanel"></div> */
  function mountSettings(el, onSave) {
    el.innerHTML = `<details class="settings"><summary>Settings: GitHub token ${token() ? "✓" : "(needed)"} · live prices ${quoteKey() ? "✓" : "(off)"}</summary>
      <div>GitHub fine-grained token for this repository with <b>Contents</b> and <b>Actions</b> set to read and write.</div>
      <input type="password" placeholder="github_pat_..." aria-label="GitHub token" data-k="tok">
      <div style="margin-top:6px">Finnhub API key for live prices (free at finnhub.io).</div>
      <input type="password" placeholder="Finnhub key" aria-label="Finnhub API key" data-k="fh">
      <button class="ghost">Save</button><div>Both are stored only in this browser.</div></details>`;
    el.querySelector("button").onclick = () => {
      const tok = el.querySelector("[data-k=tok]").value.trim(), fh = el.querySelector("[data-k=fh]").value.trim();
      if (tok) setToken(tok);
      if (fh) ls.set("fh_key", fh);
      mountSettings(el, onSave); onSave && onSave();
    };
  }

  return {load, update, returnPct, dispatch, token, setToken, canWrite, mountSettings, fmtTime, quote, quoteKey, readRepoJson,
          get trades() { return cache.trades; }};
})();
