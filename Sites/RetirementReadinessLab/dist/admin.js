const $ = selector => document.querySelector(selector);
const number = new Intl.NumberFormat('en-US');
const day = new Intl.DateTimeFormat('en-US', { month: 'short', day: 'numeric', timeZone: 'UTC' });
let selectedDays = 7;
let loadRevision = 0;

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
  });
}
$('#refresh').addEventListener('click', load);
load();
