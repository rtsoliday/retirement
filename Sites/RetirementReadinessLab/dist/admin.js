const $ = selector => document.querySelector(selector);
const number = new Intl.NumberFormat('en-US');
const day = new Intl.DateTimeFormat('en-US', { month: 'short', day: 'numeric', timeZone: 'UTC' });
let selectedDays = 7;
let loadRevision = 0;
let extraRevision = 0;
const money = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' });

function tableRows(selector, rows) {
  const body = $(selector);
  body.replaceChildren();
  for (const values of rows) {
    const tr = document.createElement('tr');
    for (const value of values) { const td = document.createElement('td'); td.textContent = value; tr.append(td); }
    body.append(tr);
  }
}
async function loadExtras() {
  const revision = ++extraRevision, days = selectedDays;
  await Promise.all(['usage', 'billing', 'mcp'].map(async section => {
    const message = $(`#${section}-status`), panel = $(`#${section}-data`);
    panel.hidden = true;
    message.textContent = section === 'usage' ? 'Loading planner usage…' : section === 'mcp' ? 'Loading assistant tool usage…' : 'Loading Stripe figures…';
    message.classList.toggle('error', false);
    try {
      const response = await fetch(`/api/admin/${section}?days=${days}`, { credentials: 'same-origin', cache: 'no-store' });
      const data = await response.json();
      if (revision !== extraRevision) return;
      if (!response.ok) throw new Error(data.error || 'Statistics unavailable. Please retry.');
      if (section === 'usage') {
        $('#simulation-starts').textContent = number.format(data.simulationStarts);
        $('#usage-period').textContent = `Last ${days} days · UTC · ${data.firstRecordedDate ? 'First recorded activity: ' + data.firstRecordedDate : 'No activity recorded yet'}`;
        chart('#starts-chart', data.series, 'starts', 'Daily simulation starts');
        tableRows('#referral-rows', data.referrals.length ? data.referrals.map(row => [row.source, number.format(row.arrivals)]) : [['No arrivals recorded yet', '—']]);
        message.textContent = 'Planner usage loaded.';
      } else if (section === 'mcp') {
        $('#mcp-calls').textContent = number.format(data.calls);
        $('#mcp-completed').textContent = number.format(data.completed);
        tableRows('#mcp-rows', data.rows.length ? data.rows.map(row => [row.tool, row.outcome, row.duration_band, number.format(row.count)]) : [['No calls recorded yet', '—', '—', '—']]);
        message.textContent = `Last ${days} days · UTC · ${data.generalAccessEnabled ? 'General access enabled' : 'Owner verification only'}`;
      } else {
        $('#paid-subscriptions').textContent = number.format(data.activeSubscriptions);
        $('#monthly-revenue').textContent = money.format(data.monthlyRecurringCents / 100);
        $('#collected-revenue').textContent = money.format(data.grossCollectedCents / 100);
        message.textContent = `${data.mode === 'test' ? 'TEST MODE — simulated payments, not real revenue.' : 'Live Stripe figures.'} Current subscriptions · payments in last ${days} days (UTC) · Fetched ${new Date().toLocaleString()}`;
      }
      panel.hidden = false;
    } catch (error) {
      if (revision !== extraRevision) return;
      message.textContent = error.message || 'Statistics unavailable. Please retry.';
      message.classList.toggle('error', true);
    }
  }));
}

function label(date) { return day.format(new Date(`${date}T00:00:00Z`)); }
function status(message, error = false) {
  $('#status').textContent = message;
  $('#status').classList.toggle('error', error);
}
function chart(target, rows, field, title) {
  const container = $(target);
  container.replaceChildren();
  container.dataset.series = field === 'uniqueIps' ? 'visitors' : 'views';
  container.setAttribute('aria-label', `${title}: ${rows.map(row => `${label(row.date)} ${row[field]}`).join(', ')}`);
  const max = Math.max(1, ...rows.map(row => row[field]));
  for (const row of rows) {
    const bar = document.createElement('div');
    bar.className = 'admin-bar';
    bar.title = `${label(row.date)}: ${number.format(row[field])}`;
    const column = document.createElement('span');
    column.style.height = `${Math.max(2, row[field] / max * 138)}px`;
    const caption = document.createElement('abbr');
    caption.textContent = String(Number(row.date.slice(-2)));
    caption.title = row.date;
    bar.append(column, caption);
    container.append(bar);
  }
}
function render(data) {
  const rows = data.series;
  $('#total-views').textContent = number.format(data.pageViews);
  $('#peak-visitors').textContent = number.format(data.peakDailyUniqueIps);
  const yesterday = new Date(Date.now() - 86400000).toISOString().slice(0, 10);
  $('#yesterday-visitors').textContent = number.format(rows.find(row => row.date === yesterday)?.uniqueIps ?? 0);
  $('#period').textContent = `${label(data.start)}–${label(rows.at(-1).date)} · UTC · Fetched ${new Date().toLocaleString()}`;
  chart('#views-chart', rows, 'pageViews', 'Daily page views');
  chart('#visitors-chart', rows, 'uniqueIps', 'Daily unique visitor IPs');
  const body = $('#daily-rows');
  body.replaceChildren();
  for (const row of [...rows].reverse()) {
    const tr = document.createElement('tr');
    for (const value of [row.date, number.format(row.pageViews), number.format(row.uniqueIps)]) {
      const td = document.createElement('td');
      td.textContent = value;
      tr.append(td);
    }
    body.append(tr);
  }
  $('#dashboard').hidden = false;
  $('#setup').hidden = true;
  status(data.pageViews || data.peakDailyUniqueIps ? 'Cloudflare traffic loaded.' : 'No traffic reported for this date range yet.');
}
async function load() {
  const revision = ++loadRevision;
  status('Loading Cloudflare traffic…');
  $('#refresh').disabled = true;
  try {
    const response = await fetch(`/api/admin/traffic?days=${selectedDays}`, { credentials: 'same-origin', cache: 'no-store' });
    const data = await response.json();
    if (revision !== loadRevision) return;
    if (!response.ok) {
      $('#dashboard').hidden = true;
      $('#setup').hidden = response.status !== 503;
      status(data.error || 'Could not load Cloudflare traffic.', true);
      return;
    }
    render(data);
  } catch {
    if (revision !== loadRevision) return;
    $('#dashboard').hidden = true;
    status('Could not connect to the analytics service.', true);
  } finally {
    if (revision === loadRevision) $('#refresh').disabled = false;
  }
}
for (const button of document.querySelectorAll('[data-days]')) {
  button.addEventListener('click', () => {
    selectedDays = Number(button.dataset.days);
    for (const option of document.querySelectorAll('[data-days]')) option.setAttribute('aria-pressed', String(option === button));
    load();
    loadExtras();
  });
}
$('#refresh').addEventListener('click', () => { load(); loadExtras(); });
load();
loadExtras();
