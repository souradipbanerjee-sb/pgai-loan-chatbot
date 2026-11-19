/**
 * Receiver for the loan chatbot's audit log.
 * Paste into the Google Sheet: Extensions > Apps Script, replace everything, set TOKEN,
 * then Deploy > New deployment > Web app (Execute as: Me, Who has access: Anyone).
 *
 * The chatbot POSTs: {token, headers: {tab: [columns]}, rows: [{table, values: [...]}, ...]}
 * Each row is appended to the tab named `table`; missing tabs are created with a header row.
 */
const TOKEN = 'PASTE-THE-SAME-TOKEN-AS-IN-STREAMLIT-SECRETS';

function doPost(e) {
  try {
    const body = JSON.parse(e.postData.contents);
    if (body.token !== TOKEN) return reply_({ ok: false, error: 'bad token' });

    const lock = LockService.getScriptLock();
    lock.waitLock(20000);                       // several app sessions may write at once
    try {
      const ss = SpreadsheetApp.getActiveSpreadsheet();
      const byTable = {};
      (body.rows || []).forEach(function (r) {
        (byTable[r.table] = byTable[r.table] || []).push(r.values);
      });
      Object.keys(byTable).forEach(function (table) {
        const headers = (body.headers || {})[table];
        if (!headers) return;
        let sh = ss.getSheetByName(table);
        if (!sh) sh = ss.insertSheet(table);
        if (sh.getLastRow() === 0) {
          sh.appendRow(headers);
          sh.setFrozenRows(1);
          sh.getRange(1, 1, 1, headers.length).setFontWeight('bold');
        }
        const rows = byTable[table].map(function (v) { return v.slice(0, headers.length); });
        sh.getRange(sh.getLastRow() + 1, 1, rows.length, headers.length).setValues(rows);
      });
    } finally {
      lock.releaseLock();
    }
    return reply_({ ok: true, rows: (body.rows || []).length });
  } catch (err) {
    return reply_({ ok: false, error: String(err) });
  }
}

function doGet() {
  return reply_({ ok: true, service: 'loan chatbot audit log' });
}

function reply_(obj) {
  return ContentService.createTextOutput(JSON.stringify(obj)).setMimeType(ContentService.MimeType.JSON);
}
